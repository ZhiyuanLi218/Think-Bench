import re
import time
import openai
import torch
import traceback
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_MESSAGE = "You are an AI assistant that helps people solve their questions."


def extract_think_content(content):
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    match = think_pattern.search(content)
    if match:
        reasoning_content = match.group(1)
        content = think_pattern.sub('', content).strip()
        return content, reasoning_content
    else:
        return content, ""


def create_client(args, timeout=3600):
    return OpenAI(api_key=args.openai_api_key, base_url=args.llm_url, timeout=timeout)


def create_messages(inputs):
    """Create message structure for API calls."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": inputs["query_input"]}
    ]


def handle_api_errors(func):
    def wrapper(inputs, args, *func_args, **func_kwargs):
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                return func(inputs, args, *func_args, **func_kwargs)
            except openai.RateLimitError as e:
                print(f'Rate limit exceeded, waiting for 60 seconds... ERROR: {e}')
                time.sleep(60)
                retries += 1
            except openai.APIConnectionError as e:
                print(f'API connection error, waiting for 10 seconds... ERROR: {e}')
                time.sleep(10)
                retries += 1
            except Exception as e:
                if 'RequestTimeOut' in str(e):
                    print(f'Timeout error, retrying in 5 seconds... ERROR: {e}')
                    time.sleep(5)
                    retries += 1
                else:
                    print(f'Unexpected error: {e}')
                    return None, None
        print(f'Max retries ({max_retries}) reached, giving up.')
        return None, None
    return wrapper


@handle_api_errors
def process_non_streaming_response(client, inputs, args, extra_body=None):
    response = client.chat.completions.create(
        model=args.model,
        messages=create_messages(inputs),
        extra_body=extra_body or {}
    )
    content = response.choices[0].message.content
    reasoning = getattr(response.choices[0].message, 'reasoning_content', None) or ""
    if reasoning:
        return content, reasoning
    return extract_think_content(content)


@handle_api_errors
def process_streaming_response(client, inputs, args, extra_body=None):
    completion = client.chat.completions.create(
        model=args.model,
        messages=create_messages(inputs),
        stream=True,
        extra_body=extra_body or {}
    )
    reasoning_content = ""
    answer_content = ""
    is_answering = False

    try:
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                elif delta.content is not None:
                    if not is_answering and delta.content:
                        is_answering = True
                    answer_content += delta.content
    except Exception as e:
        print(f'Error at input index {inputs.get("index", "unknown")}: {e}')
        return None, None

    return answer_content, reasoning_content

def deepseek(inputs, args):
    client = create_client(args)
    return process_non_streaming_response(client, inputs, args)

def qwq(inputs, args):
    client = create_client(args)
    return process_streaming_response(client, inputs, args)

def qwen3(inputs, args):
    client = create_client(args)
    return process_streaming_response(client, inputs, args, extra_body={"enable_thinking": True})

def claude(inputs, args):
    client = create_client(args)
    content, reasoning = process_non_streaming_response(client, inputs, args)
    return extract_think_content(content) if content else (None, None)

def grok3(inputs, args):
    client = create_client(args)
    return process_non_streaming_response(client, inputs, args)

def ernie(inputs, args):
    client = create_client(args)
    return process_streaming_response(client, inputs, args)

def glm(inputs, args):
    client = create_client(args)
    content, _ = process_streaming_response(client, inputs, args)
    return extract_think_content(content) if content else (None, None)

def deepseek_distill(inputs, args):
    client = create_client(args)
    return process_streaming_response(client, inputs, args)


# def qwen3_local(inputs, args):
#     try:
#         messages = create_messages(inputs)
#         text = tokenizer_qwen.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=True,
#         )
#         model_inputs = tokenizer_qwen([text], return_tensors="pt").to(model_qwen.device)

#         # Generate response
#         generated_ids = model_qwen.generate(
#             **model_inputs,
#             max_new_tokens=32768
#         )
#         output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

#         # Parse thinking and answer content
#         try:
#             # Find </think> token (151668)
#             index = len(output_ids) - output_ids[::-1].index(151668)
#         except ValueError:
#             index = 0

#         reasoning_content = tokenizer_qwen.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
#         answer_content = tokenizer_qwen.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

#         return answer_content, reasoning_content

#     except Exception as e:
#         print(f'Error processing local Qwen3 model at input index {inputs.get("index", "unknown")}: \n\n{type(e)} | {e}\n\n{traceback.format_exc()}')
#         return None, None

model_name = "Qwen/Qwen3-32B"
model_qwen_pool = {}
tokenizer_qwen_pool = {}


def qwen3_local(inputs, args):
    try:
        index = inputs.get("index", 0)
        device = f"cuda:{index % 2}" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "cuda:0" if torch.cuda.is_available() else "cpu"

        # Если модель и токенизатор ещё не инициализированы на этом устройстве
        if device not in model_qwen_pool:
            print(f"Loading model on {device} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=device,
                low_cpu_mem_usage=True,
            )
            tokenizer_qwen_pool[device] = tokenizer
            model_qwen_pool[device] = model
        else:
            tokenizer = tokenizer_qwen_pool[device]
            model = model_qwen_pool[device]

        messages = create_messages(inputs)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index_split = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index_split = 0

        reasoning_content = tokenizer.decode(output_ids[:index_split], skip_special_tokens=True).strip("\n")
        answer_content = tokenizer.decode(output_ids[index_split:], skip_special_tokens=True).strip("\n")

        return answer_content, reasoning_content

    except Exception as e:
        print(f'Error processing local Qwen3 model at input index {inputs.get("index", "unknown")}: \n\n{type(e)} | {e}\n\n{traceback.format_exc()}')
        return None, None
