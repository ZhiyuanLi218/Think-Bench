import argparse
import os
import json
import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.query_utils import claude, deepseek, deepseek_distill, ernie, glm, grok3, qwen3, qwen3_local, qwq


models_ = {
    "claude-3-7-sonnet-thinking-all": claude,
    "deepseek-reasoner": deepseek,
    "deepseek-r1-distill-qwen-1.5b": deepseek_distill, 
    "deepseek-r1-distill-qwen-7b": deepseek_distill,
    "deepseek-r1-distill-qwen-14b": deepseek_distill,
    "deepseek-r1-distill-qwen-32b": deepseek_distill,
    "ernie-x1-turbo-32k": ernie,
    "glm-z1-air": glm,
    "grok-3-mini-beta": grok3,
    "qwen3-235b-a22b": qwen3,
    "qwen3-32b": qwen3_local, ###
    "qwq-plus": qwq,
}


# read json file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# save to cache
def save_to_cache(item, model_name, cache_dir="cache"):
    cache_dir = os.path.join(cache_dir, model_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    index = item["index"]
    cache_file = os.path.join(cache_dir, f"{index}.json")
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        print(f"save to {cache_file} error: {e}")

# load all cached items
def load_cache(model_name, cache_dir="cache"):
    cache_dir = os.path.join(cache_dir, model_name)
    cached_items = []
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(cache_dir, filename)
                try:
                    item = load_json(file_path)
                    cached_items.append(item)
                except Exception as e:
                    print(f"load cache file {file_path} error: {e}")
    return cached_items

# query llm
def query_llm(inputs, args):
    model_name = args.model.lower()
    if model_name in models_:
        prediction, think_content = models_[model_name](inputs, args)
    else:
        raise ValueError(f"ERROR: {args.model} not supported. Available models: {models_.keys()}")
    return prediction, think_content 

# process single question
def process_question(item, prompt, args):
    question = item["question"]
    query_input = prompt + question if prompt else question
    prediction, think_content = query_llm({"query_input": query_input, "index": item["index"]}, args)
    item["prediction"] = prediction
    item["think_content"] = think_content
    if prediction == "" or think_content == "":
        print(f"ERROR: output seems None: {item['index']}")
    if prediction is not None and think_content is not None:
        save_to_cache(item, args.model, args.cache_dir)
    return item

# process all questions in parallel
def process_all_questions(data, prompt, args):
    cached_items = load_cache(args.model, args.cache_dir)
    cached_indices = {item["index"] for item in cached_items}
    new_data = [item for item in data if item["index"] not in cached_indices]
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        results = list(tqdm.tqdm(executor.map(lambda x: process_question(x, prompt, args), new_data), total=len(new_data)))

# save output
def save_output(results, file_name="output.json"):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=r"Think-Bench.json", help="Path to the input JSON file")
    parser.add_argument("--output_file", type=str, default=r"LLM_Output/glm-z1-air.json", help="Path to the output JSON file")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Path to the cache directory")
    parser.add_argument("--openai_api_key", type=str, required=True, help="API key")
    parser.add_argument("--llm_url", type=str, required=True, help="URL of the LLM API")
    parser.add_argument("--model", type=str, default="glm-z1-air", 
                        choices=[
                            "deepseek-reasoner",  
                            "claude-3-7-sonnet-thinking-all",
                            "grok-3-mini-beta",
                            "glm-z1-air", 
                            "qwq-plus", 
                            "qwen3-235b-a22b",
                            "qwen3-32b", ###
                            "ernie-x1-turbo-32k", 
                            "deepseek-r1-distill-qwen-1.5b", 
                            "deepseek-r1-distill-qwen-7b",
                            "deepseek-r1-distill-qwen-14b",
                            "deepseek-r1-distill-qwen-32b",
                        ], help="model to use")
    parser.add_argument("--prompt", action="store_true", default=False, help="Whether to use prompt engineering")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads.")
    args = parser.parse_args()
    # load data
    file_path = args.input_file
    data = load_json(file_path)
    # prompt engineering
    if args.prompt:
        prompt = "Answer this question with minimal thought:"
    else:
        prompt = ""
    # process all questions
    process_all_questions(data, prompt, args)
    processed_data = load_cache(args.model, args.cache_dir)
    # save output
    save_output(processed_data, args.output_file)
