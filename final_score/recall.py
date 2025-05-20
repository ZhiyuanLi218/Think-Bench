import json
import os
from collections import defaultdict
import argparse
import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import extract_json_string

# analyze single json file
def analyze_recall(json_file_path):
    """
    Analyze recall metrics for a single json file.
    
    Args:
        json_file_path (str): Path to the json file to analyze
        
    Returns:
        dict: Dictionary containing:
            - success (bool): Whether analysis was successful
            - recall (float): Overall recall score
            - logical_inference (float): Recall for logical inference steps
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
            
        gt_data = data.get('key_annotation_steps', {})
        solution1 = gt_data.get('solution1', {})
        solution2 = gt_data.get('solution2', {})
        solution3 = gt_data.get('solution3', {})
        solution4 = gt_data.get('solution4', {})

        def get_counts(solution):
            if solution is None:
                return 0, 0
            conclusions = [c for c in solution.get('logical_conclusion', []) if c.strip()]
            return len(conclusions)

        s1_conclusion_count = get_counts(solution1)
        s2_conclusion_count = get_counts(solution2)
        s3_conclusion_count = get_counts(solution3)
        s4_conclusion_count = get_counts(solution4)

        s1_count = s1_conclusion_count
        s2_count = s2_conclusion_count
        s3_count = s3_conclusion_count
        s4_count = s4_conclusion_count

        total_count = s1_count + s2_count + s3_count + s4_count
        if total_count == 0:
            raise ValueError('solution data incomplete')

        json_str = extract_json_string(data.get('valid_outputs'))
        if not json_str:
            raise ValueError('valid_outputs not found or invalid')

        steps_data = json.loads(json_str, strict=False)
        if len(steps_data) < total_count:
            pass

        current_index = 0

        # calculate recall for each solution
        # solution1
        s1_steps = steps_data[current_index:current_index + s1_count]
        current_index += s1_count
        s1_conclusion_steps = s1_steps[:s1_conclusion_count]
        matched_s1_conclusion = sum(1 for step in s1_conclusion_steps if step.get('judgment') == 'Matched')
        recall_s1_conclusion = matched_s1_conclusion / s1_conclusion_count if s1_conclusion_count else 0
        recall_s1 = matched_s1_conclusion / s1_count if s1_count else 0

        # solution2
        s2_steps = steps_data[current_index:current_index + s2_count]
        current_index += s2_count
        s2_conclusion_steps = s2_steps[:s2_conclusion_count]
        matched_s2_conclusion = sum(1 for step in s2_conclusion_steps if step.get('judgment') == 'Matched')
        recall_s2_conclusion = matched_s2_conclusion / s2_conclusion_count if s2_conclusion_count else 0
        recall_s2 = matched_s2_conclusion/ s2_count if s2_count else 0

        # solution3
        s3_steps = steps_data[current_index:current_index + s3_count]
        current_index += s3_count
        s3_conclusion_steps = s3_steps[:s3_conclusion_count]
        matched_s3_conclusion = sum(1 for step in s3_conclusion_steps if step.get('judgment') == 'Matched')
        recall_s3_conclusion = matched_s3_conclusion / s3_conclusion_count if s3_conclusion_count else 0
        recall_s3 = matched_s3_conclusion / s3_count if s3_count else 0

        # solution4
        s4_steps = steps_data[current_index:current_index + s4_count]
        s4_conclusion_steps = s4_steps[:s4_conclusion_count]
        matched_s4_conclusion = sum(1 for step in s4_conclusion_steps if step.get('judgment') == 'Matched')
        recall_s4_conclusion = matched_s4_conclusion / s4_conclusion_count if s4_conclusion_count else 0
        recall_s4 = matched_s4_conclusion / s4_count if s4_count else 0

        recalls = [(recall_s1, s1_steps, {'logical_inference': recall_s1_conclusion}),
                  (recall_s2, s2_steps, {'logical_inference': recall_s2_conclusion}),
                  (recall_s3, s3_steps, {'logical_inference': recall_s3_conclusion}),
                  (recall_s4, s4_steps, {'logical_inference': recall_s4_conclusion})]
        
        # filter out invalid recalls
        valid_recalls = [(r, s, t) for r, s, t in recalls if r is not None]
        
        if not valid_recalls:
            return {
                'success': True,
                'recall': None,
                'logical_inference': None,
                'category': None,
                'subcategory': None,
                'question_type': None
            }
            
        # choose the best solution
        final_recall, chosen_steps, type_recalls = max(valid_recalls, key=lambda x: x[0])

        data['recall'] = {
            'score': final_recall,
            'logical_inference': type_recalls['logical_inference'],
        }

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        return {
            'success': True,
            'recall': final_recall,
            'logical_inference': type_recalls['logical_inference'],
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
            - recall: List of recall scores
            - error_files: List of files with errors
            - type_metrics: Metrics by step type
            - category_metrics: Metrics by category
            - subcategory_metrics: Metrics by subcategory
            - question_type_metrics: Metrics by question type
    """
    
    results = {
        'recall': [],
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
        result = analyze_recall(json_file_path)
        
        if result['success']:
            if result['recall'] is not None:
                results['recall'].append(result['recall'])
                if result['logical_inference'] is not None:
                    results['type_metrics']['logical_inference'].append(result['logical_inference'])
                    
                # category and subcategory metrics
                if result['category']:
                    results['category_metrics'][result['category']].append(result['recall'])
                if result['subcategory']:
                    results['subcategory_metrics'][result['subcategory']].append(result['recall'])
                if result['question_type']:
                    results['question_type_metrics'][result['question_type']].append(result['recall'])
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
    
    save_dir = os.path.join(save_path, 'recall')
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
                "average_recall": round(sum(results['recall'])/len(results['recall']), 4) if results['recall'] else None
            },
            "type_metrics": {
                "logical inference": {
                    "average_recall": round(sum(results['type_metrics']['logical_inference'])/len(results['type_metrics']['logical_inference']), 4) 
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
    output_file = os.path.join(save_dir, 'recall_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    # Save error file if exists
    if all_error_files:
        error_file = os.path.join(save_dir, 'recall_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(all_error_files, f, indent=4, ensure_ascii=False)

    return model_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate recall')
    parser.add_argument('--cache_dir', type=str, 
                       default='cache/recall',
                       help='cache directory')
    parser.add_argument('--save_path', type=str,
                       default='./final_results',
                       help='output directory')
    args = parser.parse_args()
    process_all_models(args.cache_dir, args.save_path)
    print('Done')