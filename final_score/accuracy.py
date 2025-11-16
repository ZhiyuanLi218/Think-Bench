import json
import re
import os
from collections import defaultdict
import argparse
import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import extract_json_string

# analyze single json file
def analyze_accuracy(json_file_path):
    """
    Analyze accuracy metrics for a single json file.
    
    Args:
        json_file_path (str): Path to the json file to analyze
        
    Returns:
        dict: Dictionary containing:
            - success (bool): Whether analysis was successful
            - accuracy (float): Overall accuracy score
            - logical_inference (float): Accuracy for logical inference steps
            - category (str): Problem category
            - subcategory (str): Problem subcategory
            - error (str): Error message if success is False
    """
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        category = data.get('category', None)
        subcategory = data.get('subcategory', None)
        question_type = data.get('question_type', None)


        json_str = extract_json_string(data.get('valid_outputs'))
        if not json_str:
            raise ValueError('valid_outputs not found or invalid')

        steps_data = json.loads(json_str, strict=False)

        correct = int(steps_data[0].get('correct_answer') == 'true')

        data['accuracy'] = {
            'score': correct,
            'logical_inference': correct,
        }

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return {
            'success': True,
            'accuracy': correct,
            'logical_inference': correct,
            'category': category,
            'subcategory': subcategory,
            'question_type': question_type
        }
    except json.JSONDecodeError as e:
        return {'success': False, 'error': f'JSON decode error: {str(e)}'}
    except ValueError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': f'unknown error: {str(e)}'}

# process one model
def process_model_files(model_path):
    """
    Process all json files in a model directory.
    
    Args:
        model_path: Directory containing model json files
        
    Returns:
        Dictionary containing aggregated metrics:
            - accuracy: List of accuracy scores
            - error_files: List of files with errors
            - type_metrics: Metrics by step type
            - category_metrics: Metrics by category
            - subcategory_metrics: Metrics by subcategory
            - question_type_metrics: Metrics by question type
    """
    
    results = {
        'accuracy': [],
        'error_files': [],
        'type_metrics': defaultdict(list),
        'category_metrics': defaultdict(list),
        'subcategory_metrics': defaultdict(list),
        'question_type_metrics': defaultdict(list)
    }
    
    for json_file in os.listdir(model_path):
        if not json_file.endswith('.json'):
            continue
        json_file_path = os.path.join(model_path, json_file)
        result = analyze_accuracy(json_file_path)
        
        if result['success']:
            if result['accuracy'] is not None:
                results['accuracy'].append(result['accuracy'])
                if result['logical_inference'] is not None:
                    results['type_metrics']['logical_inference'].append(result['logical_inference'])
                    
                # category and subcategory metrics
                if result['category']:
                    results['category_metrics'][result['category']].append(result['accuracy'])
                if result['subcategory']:
                    results['subcategory_metrics'][result['subcategory']].append(result['accuracy'])
                if result['question_type']:
                    results['question_type_metrics'][result['question_type']].append(result['accuracy'])
        else:
            results['error_files'].append({
                'file': json_file,
                'error': result.get('error', '未知错误')
            })
            
    return results

# process all models
def process_all_models(cache_dir, save_path):
    """Process all models and save aggregated results"""
    model_stats = {}
    results_data = {}
    all_error_files = {}
    
    save_dir = os.path.join(save_path, 'accuracy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for model in tqdm.tqdm(os.listdir(cache_dir)):
        model_path = os.path.join(cache_dir, model)
        if not os.path.isdir(model_path):
            continue
            
        results = process_model_files(model_path)
        model_stats[model] = results
        
        if results['error_files']:
            all_error_files[model] = results['error_files']
        
        # Calculate averages with rounding to 4 decimal places
        model_results = {
            "overall_metrics": {
                "average_accuracy": round(sum(results['accuracy'])/len(results['accuracy']), 4) if results['accuracy'] else None
            },
            "type_metrics": {
                "logical inference": {
                    "average_accuracy": round(sum(results['type_metrics']['logical_inference'])/len(results['type_metrics']['logical_inference']), 4) 
                    if results['type_metrics']['logical_inference'] else None
                }
            },
            "category": {
                cat: round(sum(vals)/len(vals), 4) if vals else None 
                for cat, vals in results['category_metrics'].items()
            },
            "subcategory": {
                subcat: round(sum(vals)/len(vals), 4) if vals else None
                for subcat, vals in results['subcategory_metrics'].items()
            },
            "question_type": {
                qtype: round(sum(vals)/len(vals), 4) if vals else None
                for qtype, vals in results['question_type_metrics'].items()
            }
        }
        
        results_data[model] = model_results

    # Save main results 
    output_file = os.path.join(save_dir, 'accuracy_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    # Save error file if exists
    if all_error_files:
        error_file = os.path.join(save_dir, 'accuracy_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(all_error_files, f, indent=4, ensure_ascii=False)

    return model_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate accuracy')
    parser.add_argument('--cache_dir', type=str, 
                       default='cache/recall',
                       help='cache directory')
    parser.add_argument('--save_path', type=str,
                       default='./final_results',
                       help='output directory')
    args = parser.parse_args()
    process_all_models(args.cache_dir, args.save_path)
    print('Done')
