import gradio as gr
import numpy as np
import pandas as pd
from scripts.benchmarks.benchmark import (
    gather_report,
    RESULT_FIELD_ALLOCATED_GPU_MEM as _RESULT_FIELD_ALLOCATED_GPU_MEM,
    RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM as _RESULT_FIELD_PEAK_ALLOCATED_GPU_MEM,
    RESULT_FIELD_RESERVED_GPU_MEM as _RESULT_FIELD_RESERVED_GPU_MEM,
)

from typing import List, Dict
import argparse
from functools import partial

_RESULT_FIELD_TRAIN_TOKENS_PER_SEC = 'train_tokens_per_second'

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
    COL_TRAIN_TOKENS_PER_SEC: lambda x: round(x) if not pd.isna(x) else x,
    COL_TRAIN_LOSS: lambda x: round(x, 2),
    COL_TORCH_ALLOC_MEM: lambda x: round(x / 1024 ** 2),
    COL_TORCH_PEAK_MEM: lambda x: round(x / 1024 ** 2),
    COL_NUM_GPUS: lambda x: str(x),
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

DEFAULT_TOOLTIP = [
    COL_MODEL_NAME_OR_PATH,
    COL_FRAMEWORK_CONFIG,
    COL_PEFT_METHOD,
]

METRIC_RANK = 'rank'
BARPLOT_1 = {
    'x': METRIC_RANK,
    'color': COL_FRAMEWORK_CONFIG,
    'group': COL_NUM_GPUS,
    'group_title': COL_NUM_GPUS,
    'vertical': False,
}
def FORMAT_1(df: pd.DataFrame, chart: str):
    TPS = []
    for _, A in df.groupby(COL_NUM_GPUS):
        A[METRIC_RANK] = A[chart].rank(
            method='first', ascending=False
        )
        TPS.append(A)
    return pd.concat(TPS)

BARPLOT_2 = {
    'x': COL_NUM_GPUS,
    'color': COL_FRAMEWORK_CONFIG,
    'group': METRIC_RANK,
    'group_title': METRIC_RANK,
    'vertical': False,
}

def FORMAT_2(df: pd.DataFrame, chart: str, ascending: bool = True):
    A = df.groupby([
        COL_MODEL_NAME_OR_PATH, COL_FRAMEWORK_CONFIG, COL_PEFT_METHOD
    ])[chart].max().to_frame()
    A= A.loc[A[chart] > 0] # take out those wtih zerw
    A[METRIC_RANK] = A.rank(
        method='first', ascending=ascending
    )
    return df.set_index(MAIN_COLUMNS).join(
        A.drop(chart, axis=1), 
        on=[COL_MODEL_NAME_OR_PATH, COL_FRAMEWORK_CONFIG, COL_PEFT_METHOD],
        how='inner'
    ).reset_index()

CHARTS = {
    COL_TRAIN_TOKENS_PER_SEC: BARPLOT_2, 
    COL_TORCH_PEAK_MEM: BARPLOT_2,
    COL_TORCH_ALLOC_MEM: BARPLOT_2,
}

FUNCS = {
    COL_TRAIN_TOKENS_PER_SEC: partial(FORMAT_2, ascending=False),
    COL_TORCH_PEAK_MEM: FORMAT_2,
    COL_TORCH_ALLOC_MEM: FORMAT_2,
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

# for now handle by max
# - for metrics where higher is better, this is best case
# - otherwise this is worst case
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

# binds to df by closure
def update(
    model_name_or_path: str, 
    num_gpus,
    framework_config: str,
    peft_method: str,
    chart: str,
):

    # dedupe the rows by MAIN_COLUMNS
    _df = handle_duplicates(df, MAIN_COLUMNS, chart)

    # select
    _df = select_data(_df, COL_MODEL_NAME_OR_PATH, model_name_or_path)
    _df = select_data(_df, COL_NUM_GPUS, num_gpus)
    _df = select_data(_df, COL_FRAMEWORK_CONFIG, framework_config)
    _df = select_data(_df, COL_PEFT_METHOD, peft_method)

    return _df, gr.BarPlot(
        FUNCS[chart](_df, chart), 
        y=chart, label=chart,
        **CHARTS[chart],
        tooltip=[
            *DEFAULT_TOOLTIP, chart
        ]
    )

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
                    COL_TRAIN_TOKENS_PER_SEC,
                    COL_TORCH_PEAK_MEM,
                    COL_TORCH_ALLOC_MEM,
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