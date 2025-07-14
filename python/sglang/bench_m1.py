import requests
import threading
import concurrent.futures
import time
import csv
import json
import numpy as np
from queue import Queue
import logging
import os
import argparse
from logging.handlers import TimedRotatingFileHandler

def get_logger(loger_name, file_name):
    # 创建日志记录器
    logger = logging.getLogger(loger_name)
    logger.setLevel(logging.INFO)  # 设置日志级别
    # 创建日志文件夹（如果不存在）
    log_folder = 'logs'
    os.makedirs(log_folder, exist_ok=True)
    # 配置TimedRotatingFileHandler，按天分割日志文件
    log_file = os.path.join(log_folder, file_name)
    handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',  # 每天午夜分割日志
        interval=1,  # 每隔1天分割一次
        backupCount=365  # 保留7天的日志文件
    )
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 将处理器添加到记录器
    logger.addHandler(handler)
    return logger

def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text

def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text

class RequestHandler:
    def __init__(self, provider, concurrency, input_length, output_length, test_data: Queue):
        self.provider_name = provider.get("name")
        self.api_key = provider.get("api_key")
        self.base_url = provider.get("base_url")
        self.model_name = provider.get("model_name")
        self.model_category = provider.get("model_category")
        self.concurrency = concurrency
        self.input_length = input_length
        self.output_length = output_length
        self.test_data = test_data

    def _send_request(self):

        # 初始化 token 计数器与文本变量
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        content_tokens = 0
        total_tokens = 0

        # 初始化计时变量
        start_time = time.perf_counter()
        time_to_first_token = None

        # 用于记录 reasoning 与 content 部分开始与结束的时刻
        reasoning_start_time = None
        reasoning_end_time = None
        content_start_time = None
        content_end_time = None
        usage_content = ""

        try:
            # 设置请求头
            if self.api_key:
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            else:
                headers = {
                    'Content-Type': 'application/json'
                }

            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": self.test_data.get()
                    }
                ],
                "temperature": 0.6,
                "ignore_eos": True,
                "max_tokens": self.output_length,
                "stream": True,
                "stream_options": {"include_usage": True},
                "logit_bias": {"1": -100}
            }
            # 发送 POST 请求
            start_time = time.perf_counter()
            response = requests.post(
                self.base_url + "/v1/chat/completions",
                headers=headers,
                json=data,
                stream=True  # 启用流式响应
            )
            # 检查响应状态码
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        if "resource exhausted" in chunk:
                            logging.error(response.content)
                            return None
                        try:
                            chunk = remove_prefix(chunk, "data:")
                            ## First token
                            if time_to_first_token is None:
                                time_to_first_token = time.perf_counter() - start_time
                            data = json.loads(chunk)
                            # print(json.dumps(data, indent=4, ensure_ascii=False))
                            
                            usage = data["usage"]
                            choices = data["choices"]
                            if len(choices) > 0 and choices[0]["finish_reason"] is not None:
                                prompt_tokens = usage['prompt_tokens']
                                total_tokens = usage['total_tokens']
                                completion_tokens = usage['completion_tokens'] + 1
                        except json.JSONDecodeError:
                            # 忽略无法解析的 JSON 数据
                            pass
                        except Exception as e:
                            print(f"Other decode error, {e}")
            else:
                logging.info(f"请求失败，状态码: {response.status_code}")
                logging.info(response)
                return None

            end_time = time.perf_counter()
            total_time = end_time - start_time
            token_per_second = completion_tokens / total_time
            time_per_output_token = (total_time - time_to_first_token)  / \
                (completion_tokens - 1) if (completion_tokens > 1) else 0
            reasoning_time = (reasoning_end_time - reasoning_start_time) if (
                reasoning_start_time and reasoning_end_time) else 0
            content_time = (content_end_time - content_start_time) if (
                content_start_time and content_end_time) else 0

            if completion_tokens < self.output_length:
                logging.info(f"请求返回tokens不符合要求，响应tokens数: {completion_tokens}")
                logging.info(f"response:{response.text}")

            return {
                "start_time": start_time,
                "end_time": end_time,
                "prompt_tokens": prompt_tokens,
                "reasoning_tokens": reasoning_tokens,
                "content_tokens": content_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "time_to_first_token": time_to_first_token,
                "time_per_output_token": time_per_output_token,
                "reasoning_time": reasoning_time,
                "content_time": content_time,
                "total_time": total_time,
                "token_per_second": token_per_second
            }

        except Exception as e:
            logging.info(
                f"线程名: {threading.current_thread().name}, 服务商: {self.provider_name}, 并发数： {self.concurrency}, 输入tokens长度： {self.input_length}, 测试过程中发生错误：{e}")
            return None


def write_raw_results_to_csv(provider, model_category, card_num, qps, concurrency, input_length, output_length, results):
    current_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    thread = "thread" if concurrency == 1 else "threads"
    test_output_path_raw = test_output_path + "/raw"
    if not os.path.exists(test_output_path_raw):
        os.mkdir(test_output_path_raw)
    # 写入CSV文件
    with open(f"{test_output_path_raw}/{provider}_{model_category}_{card_num}card_num_{qps}qps_{input_length}in_{output_length}out_{concurrency}{thread}_{current_time}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            "Start Time",
            "End Time",
            "Prompt Tokens",
            "Reasoning Tokens",
            "Content Tokens",
            "Completion Tokens",
            "Total Tokens",
            "Time To First Token",
            "Time Per Output Token",
            "Reasoning Time",
            "Content Time",
            "Total Time",
            "Token Per Second"
        ])
        # 写入数据
        for result in results:
            writer.writerow([
                result["start_time"],
                result["end_time"],
                result["prompt_tokens"],
                result["reasoning_tokens"],
                result["content_tokens"],
                result["completion_tokens"],
                result["total_tokens"],
                result["time_to_first_token"],
                result["time_per_output_token"],
                result["reasoning_time"],
                result["content_time"],
                result["total_time"],
                result["token_per_second"]
            ])


def calculate_metrics(results, log_sw=True):
    # 计算总处理时间
    first_start = np.min(
        [result['start_time'] for result in results if result])
    last_end = np.max([result['end_time'] for result in results if result])
    e2e_total_time = last_end - first_start
    total_requests = len(results)
    error_requests = len([result for result in results if not result])
    error_rate = error_requests / total_requests if total_requests else 0  # 需要定义可容忍错误率
    # 计算qps
    qps = total_requests / e2e_total_time if e2e_total_time else 0

    # 计算单次请求时间
    all_total_times = [result['total_time'] for result in results if result]
    avg_total_time, max_total_time, min_total_time, tp50_total_time, tp80_total_time, tp90_total_time, tp99_total_time = cal_metrics(
        all_total_times)

    # 计算单次请求tokens
    all_prompt_tokens = [
        result['prompt_tokens'] for result in results if result
    ]
    avg_prompt_tokens, max_prompt_tokens, min_prompt_tokens, tp50_prompt_tokens, tp80_prompt_tokens, tp90_prompt_tokens, tp99_prompt_tokens = cal_metrics(
        all_prompt_tokens)
    all_completion_tokens = [
        result['completion_tokens'] for result in results if result
    ]
    avg_completion_tokens, max_completion_tokens, min_completion_tokens, tp50_completion_tokens, tp80_completion_tokens, tp90_completion_tokens, tp99_completion_tokens = cal_metrics(
        all_completion_tokens)

    # 计算 TPS
    all_tps = [result['token_per_second'] for result in results if result]
    avg_tps_per_req = np.sum(all_tps) / total_requests if total_requests else 0
    total_output_tokens = np.sum(all_completion_tokens)
    e2e_tps = total_output_tokens / e2e_total_time if e2e_total_time else 0

    # 计算 TTFT
    ttfts = [result['time_to_first_token'] for result in results if result]
    avg_ttft, max_ttft, min_ttft, tp50_ttft, tp80_ttft, tp90_ttft, tp99_ttft = cal_metrics(
        ttfts)

    # 计算 TPOT
    tpots = [result['time_per_output_token'] for result in results if result]
    avg_tpot, max_tpot, min_tpot, tp50_tpot, tp80_tpot, tp90_tpot, tp99_tpot = cal_metrics(
        tpots)

    if log_sw:
        pretty_log_metrics(
            avg_completion_tokens, avg_prompt_tokens, avg_total_time, avg_tpot,
            avg_tps_per_req, avg_ttft, e2e_total_time, e2e_tps, error_rate,
            error_requests, max_completion_tokens, max_prompt_tokens,
            max_total_time, max_tpot, max_ttft, min_completion_tokens,
            min_prompt_tokens, min_total_time, min_tpot, min_ttft, qps,
            total_requests, tp50_completion_tokens, tp50_prompt_tokens,
            tp50_total_time, tp50_tpot, tp50_ttft, tp80_completion_tokens,
            tp80_prompt_tokens, tp80_total_time, tp80_tpot, tp80_ttft,
            tp90_completion_tokens, tp90_prompt_tokens, tp90_total_time,
            tp90_tpot, tp90_ttft, tp99_completion_tokens, tp99_prompt_tokens,
            tp99_total_time, tp99_tpot, tp99_ttft)

    return gen_metrics_structure(
        avg_completion_tokens, avg_prompt_tokens, avg_total_time, avg_tpot,
        avg_tps_per_req, avg_ttft, e2e_total_time, e2e_tps, error_rate,
        error_requests, max_completion_tokens, max_prompt_tokens,
        max_total_time, max_tpot, max_ttft, min_completion_tokens,
        min_prompt_tokens, min_total_time, min_tpot, min_ttft, qps,
        total_requests, tp50_completion_tokens, tp50_prompt_tokens,
        tp50_total_time, tp50_tpot, tp50_ttft, tp80_completion_tokens,
        tp80_prompt_tokens, tp80_total_time, tp80_tpot, tp80_ttft,
        tp90_completion_tokens, tp90_prompt_tokens, tp90_total_time, tp90_tpot,
        tp90_ttft, tp99_completion_tokens, tp99_prompt_tokens, tp99_total_time,
        tp99_tpot, tp99_ttft)


def cal_metrics(all_total_metrics):
    tp50_metric = np.percentile(all_total_metrics, 50)
    tp80_metric = np.percentile(all_total_metrics, 80)
    tp90_metric = np.percentile(all_total_metrics, 90)
    tp99_metric = np.percentile(all_total_metrics, 99)
    max_metric = np.max(all_total_metrics)
    min_metric = np.min(all_total_metrics)
    avg_metric = np.mean(all_total_metrics)
    return avg_metric, max_metric, min_metric, tp50_metric, tp80_metric, tp90_metric, tp99_metric

def gen_metrics_structure(
        avg_completion_tokens, avg_prompt_tokens, avg_total_time, avg_tpot,
        avg_tps_per_req, avg_ttft, e2e_total_time, e2e_tps, error_rate,
        error_requests, max_completion_tokens, max_prompt_tokens,
        max_total_time, max_tpot, max_ttft, min_completion_tokens,
        min_prompt_tokens, min_total_time, min_tpot, min_ttft, qps,
        total_requests, tp50_completion_tokens, tp50_prompt_tokens,
        tp50_total_time, tp50_tpot, tp50_ttft, tp80_completion_tokens,
        tp80_prompt_tokens, tp80_total_time, tp80_tpot, tp80_ttft,
        tp90_completion_tokens, tp90_prompt_tokens, tp90_total_time, tp90_tpot,
        tp90_ttft, tp99_completion_tokens, tp99_prompt_tokens, tp99_total_time,
        tp99_tpot, tp99_ttft):
    return {
        "e2e_total_time": e2e_total_time,
        "total_requests": total_requests,
        "error_requests": error_requests,
        "error_rate": error_rate,
        "qps": qps,
        "avg_tps_per_req": avg_tps_per_req,
        "e2e_tps": e2e_tps,
        "tp50_total_time": tp50_total_time,
        "tp80_total_time": tp80_total_time,
        "tp90_total_time": tp90_total_time,
        "tp99_total_time": tp99_total_time,
        "max_total_time": max_total_time,
        "min_total_time": min_total_time,
        "avg_total_time": avg_total_time,
        "tp50_prompt_tokens": tp50_prompt_tokens,
        "tp80_prompt_tokens": tp80_prompt_tokens,
        "tp90_prompt_tokens": tp90_prompt_tokens,
        "tp99_prompt_tokens": tp99_prompt_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "min_prompt_tokens": min_prompt_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "tp50_completion_tokens": tp50_completion_tokens,
        "tp80_completion_tokens": tp80_completion_tokens,
        "tp90_completion_tokens": tp90_completion_tokens,
        "tp99_completion_tokens": tp99_completion_tokens,
        "max_completion_tokens": max_completion_tokens,
        "min_completion_tokens": min_completion_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "tp50_ttft": tp50_ttft,
        "tp80_ttft": tp80_ttft,
        "tp90_ttft": tp90_ttft,
        "tp99_ttft": tp99_ttft,
        "max_ttft": max_ttft,
        "min_ttft": min_ttft,
        "avg_ttft": avg_ttft,
        "tp50_tpot": tp50_tpot,
        "tp80_tpot": tp80_tpot,
        "tp90_tpot": tp90_tpot,
        "tp99_tpot": tp99_tpot,
        "max_tpot": max_tpot,
        "min_tpot": min_tpot,
        "avg_tpot": avg_tpot,
    }


def pretty_log_metrics(
        avg_completion_tokens, avg_prompt_tokens, avg_total_time, avg_tpot,
        avg_tps_per_req, avg_ttft, e2e_total_time, e2e_tps, error_rate,
        error_requests, max_completion_tokens, max_prompt_tokens,
        max_total_time, max_tpot, max_ttft, min_completion_tokens,
        min_prompt_tokens, min_total_time, min_tpot, min_ttft, qps,
        total_requests, tp50_completion_tokens, tp50_prompt_tokens,
        tp50_total_time, tp50_tpot, tp50_ttft, tp80_completion_tokens,
        tp80_prompt_tokens, tp80_total_time, tp80_tpot, tp80_ttft,
        tp90_completion_tokens, tp90_prompt_tokens, tp90_total_time, tp90_tpot,
        tp90_ttft, tp99_completion_tokens, tp99_prompt_tokens, tp99_total_time,
        tp99_tpot, tp99_ttft):
    util_logger.info(f"---------------------------\n")
    util_logger.info(f"E2E_TOTAL_TIME: {e2e_total_time:.4f} seconds")
    util_logger.info(f"TOTAL_REQUESTS: {total_requests}")
    util_logger.info(f"ERROR_REQUESTS: {error_requests}")
    util_logger.info(f"ERROR_RATE: {error_rate:.4f}")
    util_logger.info(f"ACTUAL_QPS: {qps} requests/second")
    util_logger.info(f"AVG_TPS_PER_REQ: {avg_tps_per_req:.4f} tokens/second")
    util_logger.info(f"E2E_TPS: {e2e_tps:.4f} tokens/second")
    util_logger.info(f"TP50_PROMPT_TOKENS: {tp50_prompt_tokens:.4f} tokens")
    util_logger.info(f"TP80_PROMPT_TOKENS: {tp80_prompt_tokens:.4f} tokens")
    util_logger.info(f"TP90_PROMPT_TOKENS: {tp90_prompt_tokens:.4f} tokens")
    util_logger.info(f"TP99_PROMPT_TOKENS: {tp99_prompt_tokens:.4f} tokens")
    util_logger.info(f"MAX_PROMPT_TOKENS: {max_prompt_tokens} tokens")
    util_logger.info(f"MIN_PROMPT_TOKENS: {min_prompt_tokens} tokens")
    util_logger.info(f"AVG_PROMPT_TOKENS: {avg_prompt_tokens:.4f} tokens")
    util_logger.info(
        f"TP50_COMPLETION_TOKENS: {tp50_completion_tokens:.4f} tokens")
    util_logger.info(
        f"TP80_COMPLETION_TOKENS: {tp80_completion_tokens:.4f} tokens")
    util_logger.info(
        f"TP90_COMPLETION_TOKENS: {tp90_completion_tokens:.4f} tokens")
    util_logger.info(
        f"TP99_COMPLETION_TOKENS: {tp99_completion_tokens:.4f} tokens")
    util_logger.info(f"MAX_COMPLETION_TOKENS: {max_completion_tokens} tokens")
    util_logger.info(f"MIN_COMPLETION_TOKENS: {min_completion_tokens} tokens")
    util_logger.info(
        f"AVG_COMPLETION_TOKENS: {avg_completion_tokens:.4f} tokens")
    util_logger.info(f"TP50_TOTAL_TIME: {tp50_total_time:.4f} seconds")
    util_logger.info(f"TP80_TOTAL_TIME: {tp80_total_time:.4f} seconds")
    util_logger.info(f"TP90_TOTAL_TIME: {tp90_total_time:.4f} seconds")
    util_logger.info(f"TP99_TOTAL_TIME: {tp99_total_time:.4f} seconds")
    util_logger.info(f"MAX_TOTAL_TIME: {max_total_time} seconds")
    util_logger.info(f"MIN_TOTAL_TIME: {min_total_time} seconds")
    util_logger.info(f"AVG_TOTAL_TIME: {avg_total_time:.4f} seconds")
    util_logger.info(f"TP50_TTFT: {tp50_ttft:.4f} seconds")
    util_logger.info(f"TP80_TTFT: {tp80_ttft:.4f} seconds")
    util_logger.info(f"TP90_TTFT: {tp90_ttft:.4f} seconds")
    util_logger.info(f"TP99_TTFT: {tp99_ttft:.4f} seconds")
    util_logger.info(f"MAX_TTFT: {max_ttft:.4f} seconds")
    util_logger.info(f"MIN_TTFT: {min_ttft:.4f} seconds")
    util_logger.info(f"AVG_TTFT: {avg_ttft:.4f} seconds")
    util_logger.info(f"TP50_TPOT: {tp50_tpot:.4f} seconds/token")
    util_logger.info(f"TP80_TPOT: {tp80_tpot:.4f} seconds/token")
    util_logger.info(f"TP90_TPOT: {tp90_tpot:.4f} seconds/token")
    util_logger.info(f"TP99_TPOT: {tp99_tpot:.4f} seconds/token")
    util_logger.info(f"MAX_TPOT: {max_tpot:.4f} seconds/token")
    util_logger.info(f"MIN_TPOT: {min_tpot:.4f} seconds/token")
    util_logger.info(f"AVG_TPOT: {avg_tpot:.4f} seconds/token")
    util_logger.info(f"---------------------------\n")


def write_summary_metrics_to_csv(provider_name, model_category, summaries):
    current_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    with open(f"{test_output_path}/summary_{provider_name}_{model_category}_{current_time}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Provider",
            "Input Length",
            "Output Length",
            "QPS"
            "Concurrency",
            "E2E Total Time",
            "Total Requests",
            "Error Requests",
            "Error Rate",
            "ACTUAL QPS",
            "QVG TPS Per Req",
            "E2E TPS",
            "TP50 Prompt Tokens",
            "TP80 Prompt Tokens",
            "TP90 Prompt Tokens",
            "TP99 Prompt Tokens",
            "MAX Prompt Tokens",
            "MIN Prompt Tokens",
            "AVG Prompt Tokens",
            "TP50 Completion Tokens",
            "TP80 Completion Tokens",
            "TP90 Completion Tokens",
            "TP99 Completion Tokens",
            "MAX Completion Tokens",
            "MIN Completion Tokens",
            "AVG Completion Tokens",
            "TP50 Total Time",
            "TP80 Total Time",
            "TP90 Total Time",
            "TP99 Total Time",
            "MAX Total Time",
            "MIN Total Time",
            "AVG Total Time",
            "TP50 TTFT",
            "TP80 TTFT",
            "TP90 TTFT",
            "TP99 TTFT",
            "MAX TTFT",
            "MIN TTFT",
            "AVG TTFT",
            "TP50 TPOT",
            "TP80 TPOT",
            "TP90 TPOT",
            "TP99 TPOT",
            "MAX TPOT",
            "MIN TPOT",
            "AVG TPOT",
        ])
        for summary in summaries:
            writer.writerow([
                summary['provider'],
                summary['input_length'],
                summary['output_length'],
                summary['qps'],
                summary['concurrency'],
                summary['metrics']["e2e_total_time"],
                summary['metrics']["total_requests"],
                summary['metrics']["error_requests"],
                summary['metrics']["error_rate"],
                summary['metrics']["qps"],
                summary['metrics']["avg_tps_per_req"],
                summary['metrics']["e2e_tps"],
                summary['metrics']["tp50_prompt_tokens"],
                summary['metrics']["tp80_prompt_tokens"],
                summary['metrics']["tp90_prompt_tokens"],
                summary['metrics']["tp99_prompt_tokens"],
                summary['metrics']["max_prompt_tokens"],
                summary['metrics']["min_prompt_tokens"],
                summary['metrics']["avg_prompt_tokens"],
                summary['metrics']["tp50_completion_tokens"],
                summary['metrics']["tp80_completion_tokens"],
                summary['metrics']["tp90_completion_tokens"],
                summary['metrics']["tp99_completion_tokens"],
                summary['metrics']["max_completion_tokens"],
                summary['metrics']["min_completion_tokens"],
                summary['metrics']["avg_completion_tokens"],
                summary['metrics']["tp50_total_time"],
                summary['metrics']["tp80_total_time"],
                summary['metrics']["tp90_total_time"],
                summary['metrics']["tp99_total_time"],
                summary['metrics']["max_total_time"],
                summary['metrics']["min_total_time"],
                summary['metrics']["avg_total_time"],
                summary['metrics']["tp50_ttft"],
                summary['metrics']["tp80_ttft"],
                summary['metrics']["tp90_ttft"],
                summary['metrics']["tp99_ttft"],
                summary['metrics']["max_ttft"],
                summary['metrics']["min_ttft"],
                summary['metrics']["avg_ttft"],
                summary['metrics']["tp50_tpot"],
                summary['metrics']["tp80_tpot"],
                summary['metrics']["tp90_tpot"],
                summary['metrics']["tp99_tpot"],
                summary['metrics']["max_tpot"],
                summary['metrics']["min_tpot"],
                summary['metrics']["avg_tpot"],
            ])


def test_concurrent_performance(provider, card_num, qps = 1, concurrency=1, task_len=320, input_length=256, output_length=256):
    test_data = Queue()
    with open(f"{test_input_path}/{test_input_name}.json", "r") as f:
        for item in json.load(f):
            test_data.put(item["input"])
    futures = []
    handler = RequestHandler(provider, concurrency,
                             input_length, output_length, test_data)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor._max_workers = concurrency
        # 以550ms为基线TTFT_min（8k最长序列max-model-len）
        # 最适batch组合e_bs=ROUNDDOWN(8k/输入序列4k)=2,  组成8k序列运行
        # 爬坡速率=e_bs*(1/TTFT_min)= 2 * (1/0.55) = 3.636  即为prefill的QPS
        # 最优爬坡方式： 以TTFT_min为间隔时间，每次发两个请求
        # growth_rate = ( math.floor( 8192 / input_length ) + 1 ) * 2  # 爬坡使用
        growth_rate = 1

        if growth_rate == 0:
            for _ in range(task_len):
                future = executor.submit(handler._send_request)
                futures.append(future)
        else:
            interval = 1 / qps
            logging.info(f"爬坡方式: 间隔时间 {interval} s; 每次新增 {growth_rate} 个请求 。")
            max_concurrency = 0
            loop = 1
            while task_len > 0 and threading.active_count() - 1 < concurrency:
                start_time = time.perf_counter()
                # current_count = threading.active_count() - 1
                # increment = loop * growth_rate - current_count
                # if increment > growth_rate:
                #     loop = loop - increment / 2
                #     logging.warning(f"decode处理较快，已有请求任务提前完成，prefill处理较慢.")
                #
                #     time_step=0.001
                #     if increment > growth_rate and executor._work_queue.qsize() == 0:
                #         interval = interval - time_step
                #         logging.info(f"当前并发{threading.active_count() -1}, 历史最大并发 {max_concurrency}; 减少爬坡间隔时间 {time_step}s 至 :{interval}")

                for __ in range( min(growth_rate, task_len) ):
                    future = executor.submit(handler._send_request)
                    futures.append(future)

                #两次间隔的时间
                elapsed = time.perf_counter() - start_time
                sleep_time = interval - elapsed
                if elapsed < interval:
                    time.sleep( sleep_time )

                loop += 1
                task_len -= growth_rate
                logging.info(
                    f"active connection: {threading.active_count() - 1}; pending:{executor._work_queue.qsize()}; 剩余待提交任务数:{task_len}")

                max_concurrency = max( max_concurrency, threading.active_count() -1 )

            # if concurrency > max_concurrency :
            #     logging.info(f"爬坡结束，已探测系统最大并发，在输入{input_length}-输出{output_length}下, 系统最优处理最大并发数为{max_concurrency}")
            # else :
            #     logging.info(f"爬坡结束，未探测到系统在输入{input_length}-输出{output_length}下最优处理的最大并发数，当前并发为{concurrency}")

            if task_len > 0 :
                for _ in range(task_len):
                    future = executor.submit(handler._send_request)
                    futures.append(future)

        while executor._work_queue.qsize():
            time.sleep(5)
            logging.info(f"active connection: {threading.active_count()-1}; pending:{executor._work_queue.qsize()}")

    # 等待所有任务完成
    logging.info(f"任务提交完成，等待所有请求结束")
    concurrent.futures.wait(futures)
    logging.info(f"所有请求已完成")

    results = [future.result()
               for future in futures if future.result() is not None]
    #write_raw_results_to_csv(provider.get("name"), provider.get(
    #    "model_category"), card_num, qps, concurrency, input_length, output_length, results)
    metrics = calculate_metrics(results)
    return metrics


# 使用示例
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Benchmark settings')
    parser.add_argument('--output_name', type=str, default='default', help='output name')
    parser.add_argument('--work_dir', type=str, default='/data/ABench/M1test/', help='work directory path')
    parser.add_argument('--host_ip', type=str, default='127.0.0.1', help='host ip')
    parser.add_argument('--port', type=int, default=8000, help='port')
    parser.add_argument('--model_name', type=str, default='Deepseek-R1', help='model name')
    parser.add_argument('--model_category', type=str, default='deepseek-r1', help='model category')
    parser.add_argument('--exp_name', type=str, default='sglang-Deepseek-R1', help='exp name')
    parser.add_argument('--concurrency', type=str, default='390', help='concurrency, comma-separated list of integers')
    parser.add_argument('--qps', type=str, default='2.4', help='qps, comma-separated list of floats')
    parser.add_argument('--input_length', type=str, default='4096', help='input length, comma-separated list of integers')
    parser.add_argument('--output_length', type=str, default='1536', help='output length, comma-separated list of integers')
    parser.add_argument('--task_len', type=int, default=4096, help='task length')
    parser.add_argument('--card_num', type=int, default=48, help='card number')
    parser.add_argument('--test_input_name', type=str, default='4096_4096', help='test input name')

    args = parser.parse_args()
    args.work_dir = os.path.abspath(args.work_dir)
    concurrencies = [int(c) for c in args.concurrency.split(',')]
    qpss = [float(q) for q in args.qps.split(',')]
    input_lengths = [int(l) for l in args.input_length.split(',')]
    output_lengths = [int(l) for l in args.output_length.split(',')]
    assert len(input_lengths) == len(output_lengths)
    input_output_lengths = list(zip(input_lengths, output_lengths))

    util_logger = get_logger('util_logger', 'scene_benchmark.log')
    test_input_path = os.path.join(args.work_dir, "data_input")
    test_input_name = args.test_input_name
    test_output_path = os.path.join(args.work_dir, "data_output", args.output_name)
    assert os.path.exists(os.path.join(test_input_path, f"{test_input_name}.json")), \
        f"input file {test_input_name}.json not exists"
    assert len(set(input_lengths)) == 1, "input length must be the same"
    assert str(input_lengths[0]) == test_input_name.split('_')[0], "input length must keep same with test_input_name"

    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
    logfile_dir = os.path.join(args.work_dir, "logs", args.output_name)
    if not os.path.exists(logfile_dir):
        os.makedirs(logfile_dir)
    logfile = os.path.join(logfile_dir, "benchmark.log")
    logging.basicConfig(filename=logfile,
                        level=logging.INFO,
                        format="%(asctime)s-%(message)s")
    # 定义各服务商的配置
    providers = [
        {
            "name": f"args.exp_name",
            "api_key": "",
            "base_url": f"http://{args.host_ip}:{args.port}",
            "model_name": f"{args.model_name}",
            "model_category": f"{args.model_category}"
        },
    ]
    # 循环对每个服务商进行测试
    for provider in providers:
        all_metrics = []
        provider_name = provider.get("name")
        model_category = provider.get("model_category")
        for concurrency in concurrencies:
            for qps in qpss:
                for input_length, output_length in input_output_lengths:
                    logging.info(f"\n---------------------------")
                    logging.info(f"开始测试服务商：{provider_name}")
                    logging.info(f"模型类型：{model_category}")
                    logging.info(f"并发数： {concurrency}")
                    logging.info(f"QPS： {qps}")
                    logging.info(f"输入tokens长度： {input_length}")
                    logging.info(f"输出tokens长度： {output_length}")

                    metrics = test_concurrent_performance(
                        provider, args.card_num, qps, concurrency, args.task_len, input_length, output_length)

                    all_metrics.append({
                        "provider": provider_name,
                        "input_length": input_length,
                        "output_length": output_length,
                        "qps": qps,
                        'concurrency': concurrency,
                        "metrics": metrics
                    })
                if len(qpss) > 1:
                    time.sleep(60)

            #write_summary_metrics_to_csv(
            #    provider_name, model_category, all_metrics)

