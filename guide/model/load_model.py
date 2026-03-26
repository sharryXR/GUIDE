import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from accelerate import Accelerator

class LocalQwen2_5_VL_32B:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-32B-Instruct", temperature=1.0, local_path=None, cache_dir=None):
        self.accelerator = Accelerator()
        pretrained_path = local_path if local_path else model_name
        load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}  
        if cache_dir and not local_path:
            load_kwargs["cache_dir"] = cache_dir

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_path, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(pretrained_path, use_fast=True)  # 使用快速处理器
        self.temperature = temperature
        # # 检查是否有多个 GPU 并使用 DataParallel 来并行化模型
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     self.model = torch.nn.DataParallel(self.model)  # 使用 DataParallel
        #使用 accelerate 来处理多 GPU
        self.model = self.accelerator.prepare(self.model)

        # # 检查可用设备，优先 MPS，其次 CUDA，最后 CPU
        # if torch.backends.mps.is_available():
        #     self.device = "mps"
        # elif torch.cuda.is_available():
        #     self.device = "cuda"
        # else:
        #     self.device = "cpu"
        # self.model.to(self.device)  # 确保模型在正确设备上

    def __call__(self, messages, frequency_penalty=None, max_tokens=128):
        # 转换 messages 格式以适配 Qwen2.5-VL
        converted_messages = []
        for msg in messages:
            converted_content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    converted_content.append(item)
                elif item["type"] == "image_url":
                    # 将 image_url 转换为 image 格式
                    converted_content.append({"type": "image", "image": item["image_url"]["url"]})
            converted_messages.append({"role": msg["role"], "content": converted_content})

        # 处理输入
        text = self.processor.apply_chat_template(
            converted_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(converted_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # inputs = inputs.to(self.device)
        inputs = self.accelerator.prepare(inputs)  # 使用 accelerate 来处理多 GPU
        
        
        # 设置 repetition_penalty（映射 frequency_penalty）
        repetition_penalty = frequency_penalty if frequency_penalty is not None else 1.0

        # 生成输出
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # 支持 max_tokens 参数
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                repetition_penalty=repetition_penalty
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # 封装成response类，output_text[0] = response.content
        response = type("Response", (object,), {})()
        response.content = output_text[0]
        # print("response", response)
        # print("response.content", response.content)             
        return response  # 返回字符串

class LocalQwen2_5_VL_7B:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", temperature=1.0, local_path=None, cache_dir=None):
        pretrained_path = local_path if local_path else model_name
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}  # 使用 bfloat16 节省内存
        if cache_dir and not local_path:
            load_kwargs["cache_dir"] = cache_dir

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_path, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(pretrained_path, use_fast=True)  # 使用快速处理器
        self.temperature = temperature
        # 检查可用设备，优先 MPS，其次 CUDA，最后 CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)  # 确保模型在正确设备上

    def __call__(self, messages, frequency_penalty=None, max_tokens=128):
        # 转换 messages 格式以适配 Qwen2.5-VL
        converted_messages = []
        for msg in messages:
            converted_content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    converted_content.append(item)
                elif item["type"] == "image_url":
                    # 将 image_url 转换为 image 格式
                    converted_content.append({"type": "image", "image": item["image_url"]["url"]})
            converted_messages.append({"role": msg["role"], "content": converted_content})

        # 处理输入
        text = self.processor.apply_chat_template(
            converted_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(converted_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 设置 repetition_penalty（映射 frequency_penalty）
        repetition_penalty = frequency_penalty if frequency_penalty is not None else 1.0

        # 生成输出
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # 支持 max_tokens 参数
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                repetition_penalty=repetition_penalty
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # 封装成response类，output_text[0] = response.content
        response = type("Response", (object,), {})()
        response.content = output_text[0]
        # print("response", response)
        # print("response.content", response.content)             
        return response  # 返回字符串


class LocalQwen2_5_VL_3B:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", temperature=1.0, local_path=None, cache_dir=None):
        pretrained_path = local_path if local_path else model_name
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}  # 使用 bfloat16 节省内存
        if cache_dir and not local_path:
            load_kwargs["cache_dir"] = cache_dir

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_path, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(pretrained_path, use_fast=True)  # 使用快速处理器
        self.temperature = temperature
        # 检查可用设备，优先 MPS，其次 CUDA，最后 CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)  # 确保模型在正确设备上

    def __call__(self, messages, frequency_penalty=None, max_tokens=128):
        # 转换 messages 格式以适配 Qwen2.5-VL
        converted_messages = []
        for msg in messages:
            converted_content = []
            for item in msg["content"]:
                if item["type"] == "text":
                    converted_content.append(item)
                elif item["type"] == "image_url":
                    # 将 image_url 转换为 image 格式
                    converted_content.append({"type": "image", "image": item["image_url"]["url"]})
            converted_messages.append({"role": msg["role"], "content": converted_content})

        # 处理输入
        text = self.processor.apply_chat_template(
            converted_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(converted_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 设置 repetition_penalty（映射 frequency_penalty）
        repetition_penalty = frequency_penalty if frequency_penalty is not None else 1.0

        # 生成输出
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # 支持 max_tokens 参数
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                repetition_penalty=repetition_penalty
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # 封装成response类，output_text[0] = response.content
        response = type("Response", (object,), {})()
        response.content = output_text[0]
        # print("response", response)
        # print("response.content", response.content)             
        return response  # 返回字符串


def load_model(model_name,temperature,local_path=None, cache_dir=None):
    """
    加载模型
    :param model_name: 模型名称
    :return: 模型实例
    """
    if model_name == "Qwen/Qwen2.5-VL-32B-Instruct":
        return LocalQwen2_5_VL_32B(model_name=model_name, temperature=temperature,local_path=local_path, cache_dir=cache_dir)
    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        return LocalQwen2_5_VL_7B(model_name=model_name, temperature=temperature,local_path=local_path, cache_dir=cache_dir)
    elif model_name == "Qwen/Qwen2.5-VL-3B-Instruct":
        return LocalQwen2_5_VL_3B(model_name=model_name, temperature=temperature,local_path=local_path, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    

