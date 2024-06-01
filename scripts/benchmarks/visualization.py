import gradio as gr
import numpy as np
import pandas as pd
from scripts.benchmarks.benchmark import (
    gather_report,
    RESULT_FIELD_ALLOCATED_GPU_MEM as _RESULT_FIELD_ALLOCATED_GPU_MEM,
    RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM as _RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM,
    RESULT_FIELD_RESERVED_GPU_MEM as _RESULT_FIELD_RESERVED_GPU_MEM,
)

_RESULT_FIELD_TRAIN_TOKENS_PER_SEC = 'train_tokens_per_second'

from typing import List, Dict
import argparse

COL_TORCH_ALLOC_MEM = 'mem_alloc'
COL_TORCH_PEAK_MEM = 'mem_peak'
COL_NVIDIA_SMI_MEN = 'nvidia_smi'
COL_MODEL_NAME_OR_PATH = 'model_name_or_path'
COL_NUM_GPUS = 'num_gpus'
COL_FRAMEWORK_CONFIG = 'framework_config'
COL_PEFT_METHOD = 'peft_method'
COL_ERROR_MESSAGES = 'error_messages'
COL_TRAIN_TOKENS_PER_SEC = 'tr_toks_per_sec'
COL_TRAIN_LOSS = 'train_loss'

COL_RENAMES = {
    _RESULT_FIELD_ALLOCATED_GPU_MEM: COL_TORCH_ALLOC_MEM,
    _RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM: COL_TORCH_PEAK_MEM,
    _RESULT_FIELD_RESERVED_GPU_MEM: COL_NVIDIA_SMI_MEN,
    _RESULT_FIELD_TRAIN_TOKENS_PER_SEC: COL_TRAIN_TOKENS_PER_SEC,
}

COL_FORMAT = {
    COL_TRAIN_LOSS: lambda x: round(x, 2),
    COL_TORCH_ALLOC_MEM: lambda x: round(x / 1024 ** 2),
    COL_TORCH_PEAK_MEM: lambda x: round(x / 1024 ** 2),
}

MAIN_COLUMNS = [
    COL_MODEL_NAME_OR_PATH,
    COL_NUM_GPUS,
    COL_FRAMEWORK_CONFIG,
    COL_PEFT_METHOD,
]

# to replace nan
DEFAULT_VALS = {
    COL_PEFT_METHOD: 'none'
}

METRIC_THROUGHPUT = 'rank'
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
    'vertical': False,
}

CHARTS = {
    COL_TRAIN_TOKENS_PER_SEC: BARPLOT_1
}

def fetch_data(
    result_dirs: List[str], columns: List[str] = None,
    renames: Dict = None
):
    if renames is None:
        renames = {}

    df, _ = gather_report(result_dirs, raw=True)
    if COL_ERROR_MESSAGES in df.columns:
        df[COL_ERROR_MESSAGES] = df[COL_ERROR_MESSAGES].isna()
    if columns is not None:
        df = df[[x for x in columns if x in df.columns]]

    # handle renames
    df = df.rename(columns=renames)
    
    # handle formatting
    for k, func in COL_FORMAT.items():
        df[k] = df[k].apply(func)

    # replace defaults
    for k, default in DEFAULT_VALS.items():
        df.loc[df[k].isna(), k]  = default
    return df

# def handle_duplicates(df: pd.DataFrame, columns: List[str]):
#     return df.drop_duplicates(columns)

# for now handle by max
def handle_duplicates(
    df: pd.DataFrame, columns: List[str],
    target_column: str, 
):
    DFS = []
    for _, A in df.groupby(columns):
        ind = A[target_column].to_numpy().argmax() # by max now
        DFS.append(A.iloc[ind:ind+1]) # to get a dataframe
    return pd.concat(DFS).sort_index()

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


# SCRIPT starts here
parser = argparse.ArgumentParser(
    prog="Visualization Script",
    description="This script runs a Gradio based visualizer",
)

parser.add_argument(
    "--result_dirs",
    nargs="+",
    help="list of result dirs to pull data from",
    default=['benchmark_outputs4', 'benchmark_outputs5']
)
parser.add_argument(
    "--server_name", help="server name to host on", default='localhost'
)
parser.add_argument(
    "--port", type=int, help="port to listen on", default=7860
)
args = parser.parse_args()

df: pd.DataFrame = None
def refresh_data():
    global df
    df = fetch_data(
        args.result_dirs, columns = [
            *MAIN_COLUMNS,
            _RESULT_FIELD_TRAIN_TOKENS_PER_SEC,
            COL_TRAIN_LOSS,
            COL_TORCH_ALLOC_MEM,
            _RESULT_FIELD_ALLOCATED_GPU_MEM,
            _RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM,
            _RESULT_FIELD_RESERVED_GPU_MEM,
        ],
        renames=COL_RENAMES,
    )

    # dedupe the rows by MAIN_COLUMNS
    df = handle_duplicates(df, MAIN_COLUMNS, COL_TRAIN_TOKENS_PER_SEC)

# binds to df by closure
def update(
    model_name_or_path: str, 
    num_gpus,
    framework_config: str,
    peft_method: str,
    chart: str,
):
    _df = df
    _df = select_data(_df, COL_MODEL_NAME_OR_PATH, model_name_or_path)
    _df = select_data(_df, COL_NUM_GPUS, num_gpus)
    _df = select_data(_df, COL_FRAMEWORK_CONFIG, framework_config)
    _df = select_data(_df, COL_PEFT_METHOD, peft_method)

    TPS = []
    for _, A in _df.groupby(COL_NUM_GPUS):
        A[METRIC_THROUGHPUT] = A[chart].rank(
            method='first', ascending=False
        )
        TPS.append(A)

    return _df, gr.BarPlot(pd.concat(TPS), **CHARTS[chart])

refresh_data()
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            mnop = create_dropdown(df, COL_MODEL_NAME_OR_PATH)
            ng = create_dropdown(df, COL_NUM_GPUS)
            fc = create_dropdown(df, COL_FRAMEWORK_CONFIG)
            pm = create_dropdown(df, COL_PEFT_METHOD)

        with gr.Column():
            chart = gr.Dropdown(
                choices=[
                    COL_TRAIN_TOKENS_PER_SEC
                ],
                value=COL_TRAIN_TOKENS_PER_SEC,
                label="Graph",
            )
            bar1 = gr.BarPlot()
    dataframe = gr.Dataframe(
        label="Benchmark Results", 
        value=df,
        interactive=False
    )
    btn = gr.Button('Display')
    btn.click(
        fn=update, 
        inputs=[mnop, ng, fc, pm, chart], 
        outputs=[dataframe, bar1]
    )
    demo.load(
        fn=update, 
        inputs=[mnop, ng, fc, pm, chart], 
        outputs=[dataframe, bar1]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name, 
        server_port=args.port
    )