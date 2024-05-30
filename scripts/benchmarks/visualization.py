import gradio as gr
import numpy as np
import pandas as pd
from scripts.benchmarks.benchmark import (
    gather_report,
    RESULT_FIELD_ALLOCATED_GPU_MEM,
    RESULT_FIELD_RESERVED_GPU_MEM,
    RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM,
)
from typing import List
import argparse

COL_MODEL_NAME_OR_PATH = 'model_name_or_path'
COL_NUM_GPUS = 'num_gpus'
COL_FRAMEWORK_CONFIG = 'framework_config'
COL_PEFT_METHOD = 'peft_method'
COL_ERROR_MESSAGES = 'error_messages'
COL_TRAIN_TOKENS_PER_SEC = 'train_tokens_per_second'
COL_TRAIN_LOSS = 'train_loss'

METRIC_THROUGHPUT = 'throughput'
BARPLOT_1 = {
    'y': COL_TRAIN_TOKENS_PER_SEC,
    'x': METRIC_THROUGHPUT,
    'color': COL_FRAMEWORK_CONFIG,
    'group': COL_NUM_GPUS,
    'title': COL_TRAIN_TOKENS_PER_SEC,
    'group_title': COL_NUM_GPUS,
    'tooltip': [
        COL_MODEL_NAME_OR_PATH,
        COL_FRAMEWORK_CONFIG,
        COL_PEFT_METHOD,
        COL_TRAIN_TOKENS_PER_SEC,
    ],
    'label': COL_TRAIN_TOKENS_PER_SEC,
}

def fetch_data(result_dirs: List[str], columns: List[str] = None):
    df, _ = gather_report(result_dirs, raw=True)
    if COL_ERROR_MESSAGES in df.columns:
        df[COL_ERROR_MESSAGES] = df[COL_ERROR_MESSAGES].isna()
    if columns is not None:
        df = df[[x for x in columns if x in df.columns]]
    return df

# values = sorted(df[column_name].unique().tolist())
def create_dropdown(df: pd.DataFrame, column_name: str):
    values = sorted(df[column_name].unique().tolist())
    return gr.Dropdown(
        values, 
        multiselect=True, 
        label=column_name,
        value=values,
    )
    
def select_data(df: pd.DataFrame, column_name: str, values: List[str]):
    if len(values) == 0:
        return df
    return df.query(f'{column_name} in @values')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Visualization Script",
        description="This script runs a Gradio based visualizer",
    )

    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="list of result dirs to pull data from",
    )
    parser.add_argument(
        "--server_name", help="server name to host on", default='localhost'
    )
    parser.add_argument(
        "--port", type=int, help="port to listen on", default=7860
    )
    args = parser.parse_args()

    MAIN_COLUMNS = [
        COL_MODEL_NAME_OR_PATH,
        COL_NUM_GPUS,
        COL_FRAMEWORK_CONFIG,
        COL_PEFT_METHOD,
    ]

    df = fetch_data(
        args.result_dirs, columns = [
            *MAIN_COLUMNS,
            COL_TRAIN_TOKENS_PER_SEC,
            COL_TRAIN_LOSS,
            RESULT_FIELD_ALLOCATED_GPU_MEM,
            RESULT_FIELD_RESERVED_GPU_MEM,
            RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM,
        ]
    )
    # laziness
    df = df.drop_duplicates(MAIN_COLUMNS)

    # binds to df by closure
    def update(
        model_name_or_path: str, 
        num_gpus,
        framework_config: str,
        peft_method: str,
    ):
        _df = df
        _df = select_data(_df, COL_MODEL_NAME_OR_PATH, model_name_or_path)
        _df = select_data(_df, COL_NUM_GPUS, num_gpus)
        _df = select_data(_df, COL_FRAMEWORK_CONFIG, framework_config)
        _df = select_data(_df, COL_PEFT_METHOD, peft_method)

        TPS = []
        for _, A in _df.groupby(COL_NUM_GPUS):
            A[METRIC_THROUGHPUT] = A[COL_TRAIN_TOKENS_PER_SEC].rank(
                method='first', ascending=False
            )
            TPS.append(A)
        
        return _df, pd.concat(TPS)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                mnop = create_dropdown(df, COL_MODEL_NAME_OR_PATH)
                ng = create_dropdown(df, COL_NUM_GPUS)
                fc = create_dropdown(df, COL_FRAMEWORK_CONFIG)
                pm = create_dropdown(df, COL_PEFT_METHOD)

            bar1 = gr.BarPlot(
                **BARPLOT_1,
                vertical=False,
            )
        dataframe = gr.Dataframe(
            label="Benchmark Results", 
            value=df,
            interactive=False
        )
        btn = gr.Button('Display')
        btn.click(fn=update, inputs=[mnop, ng, fc, pm], outputs=[dataframe, bar1])

    demo.launch(
        server_name=args.server_name, 
        server_port=args.port
    )