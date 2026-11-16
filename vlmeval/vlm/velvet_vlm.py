import os

import torch
from PIL import Image

from .. import velvet_prompts
from .base import BaseModel


class VelvetVLM(BaseModel):
    """Wrapper for the `velvet_vlm` connector checkpoints (SigLIP2 vision tower +
    MLP projector + Mistral language model), loaded via `trust_remote_code=True`
    from a local HF-export directory (see `modeling_velvet_vlm.py` in the checkpoint).
    """

    print(
        f"RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"DEVICE_COUNT={torch.cuda.device_count()}",
        flush=True,
    )

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='AlmawaveLabs/Velvet2-3B-Vision', **kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16
        ).eval()

        self.model = self.model.to("cuda")

        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    # Datasets of the training evaluation get its prompts (velvet_prompts.build_training_question), with every
    # image of the row in the prompt's order; the others keep VLMEvalKit's prompt with the SmolVLM-style rewrites.
    def use_custom_prompt(self, dataset):
        return velvet_prompts.uses_training_prompt(dataset) and self._is_multiview()

    def build_prompt(self, line, dataset=None):
        images = self.dump_image(line, dataset)
        images = images if isinstance(images, list) else [images]
        if dataset in velvet_prompts.TRAINING_MCQ and velvet_prompts.has_missing_option(line):
            if not hasattr(self, '_raw_options'):
                self._raw_options = {}
            if dataset not in self._raw_options:
                from ..smp import LMUDataRoot
                self._raw_options[dataset] = velvet_prompts.read_raw_options(
                    os.path.join(LMUDataRoot(), f'{dataset}.tsv'))
            line = velvet_prompts.restore_options(line, self._raw_options[dataset].get(str(line['index']), {}))
        question = velvet_prompts.build_training_question(line, dataset)
        return velvet_prompts.training_messages(question, images, dataset)

    def _is_multiview(self):
        return getattr(self.processor, '_is_multiview', lambda: False)()

    def generate_inner(self, message, dataset=None):
        if self.use_custom_prompt(dataset):
            # Text and images in the message's order, every image: the chat template writes "<image>\n" per image.
            content = [{'type': 'image'} if m['type'] == 'image' else {'type': 'text', 'text': m['value']}
                       for m in message]
            images = [Image.open(m['value']).convert('RGB') for m in message if m['type'] == 'image']
            chat = [{'role': 'user', 'content': content}]
            return self._generate(chat, images, dataset)
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        message_for_prompt = "You are Velvet-2-3B, a Large Language Model (LLM) created by Almawave Labs, an Italian company headquartered in Rome. You are an helpful assistant. Your goal is to provide clear, accurate, and useful and polite answers to the user"
        prompt = self.build_prompt_for_dataset(prompt, dataset)
        prompt = message_for_prompt + "\n" + prompt
        image = Image.open(image_path).convert('RGB')

        # Multi-view checkpoints need an explicit image content part so the chat template emits
        # the literal "<image>" marker that processing_velvet_vlm.py._call_multiview() counts;
        # legacy (fixed placeholder) checkpoints keep the plain string content as before.
        if getattr(self.processor, '_is_multiview', lambda: False)():
            chat = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        else:
            chat = [{'role': 'user', 'content': prompt}]
        return self._generate(chat, [image], dataset)

    def _generate(self, chat, images, dataset):
        # bos_token is already inserted by the chat template, so tokenize without adding it again
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # text_inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        # vision_inputs = self.processor(images=[image], return_tensors='pt')

        # Multi-view exports take the benchmark's own image budget, as in the training evaluation.
        processor_kwargs = {}
        overrides = velvet_prompts.vision_overrides_for(dataset)
        if overrides and getattr(self.processor, '_is_multiview', lambda: False)():
            processor_kwargs['vision_processing_overrides'] = overrides

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors='pt',
            **processor_kwargs,
        )

        device = self.model.device
        generate_kwargs = dict(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
        )
        if getattr(self.processor, '_is_multiview', lambda: False)():
            # vision_layout='views': pixel_values must stay uint8 (casting to the model dtype
            # breaks modeling_velvet_vlm.py's uint8 check), and generate() needs the view/token
            # bookkeeping tensors too, or it thinks no images were given.
            generate_kwargs['pixel_values'] = inputs['pixel_values'].to(device)
            for key in ('view_grid_hw', 'num_views_per_image', 'num_tokens_per_image', 'images_per_sample'):
                if key in inputs:
                    generate_kwargs[key] = inputs[key].to(device)
        else:
            generate_kwargs['pixel_values'] = inputs['pixel_values'].to(device=device, dtype=self.model.dtype)
            # NaFlex SigLIP2 vision towers additionally require these two tensors
            if 'pixel_attention_mask' in inputs:
                generate_kwargs['pixel_attention_mask'] = inputs['pixel_attention_mask'].to(device)
            if 'spatial_shapes' in inputs:
                generate_kwargs['spatial_shapes'] = inputs['spatial_shapes'].to(device)

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs, **self.kwargs)

        # NOTE: this model's generate() builds inputs_embeds internally (no input_ids
        # given to the underlying LM), so the returned ids are the *new* tokens only,
        # not prompt+continuation like most HF models - do not slice off the prompt length.
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response.strip()

    # The prompt rewrites live in vlmeval/velvet_prompts.py so the vLLM adapter shares them.
    def build_prompt_for_dataset(self, prompt, dataset=None):
        return velvet_prompts.build_prompt_for_dataset(prompt, dataset)

    def build_prompt_default(self, prompt, add_brief=False, add_yes_or_no=False):
        return velvet_prompts.build_prompt_default(prompt, add_brief, add_yes_or_no)

    def build_prompt_puremcq(self, prompt):
        return velvet_prompts.build_prompt_puremcq(prompt)

    def build_prompt_mmbench(self, prompt):
        return velvet_prompts.build_prompt_mmbench(prompt)

    def build_prompt_mmmu(self, prompt):
        return velvet_prompts.build_prompt_mmmu(prompt)

    def build_prompt_mathvista(self, prompt):
        return velvet_prompts.build_prompt_mathvista(prompt)

    def build_prompt_chartqa(self, prompt):
        return velvet_prompts.build_prompt_chartqa(prompt)

    def build_prompt_docvqa(self, prompt):
        return velvet_prompts.build_prompt_docvqa(prompt)

    def build_prompt_textvqa(self, prompt):
        return velvet_prompts.build_prompt_textvqa(prompt)
