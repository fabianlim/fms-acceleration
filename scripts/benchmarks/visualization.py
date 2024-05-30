import gradio as gr
import numpy as np
import pandas as pd

from scripts.benchmarks.benchmark import gather_report

remove_columns = [
    'before_init_mem_cpu', 
    'before_init_mem_gpu', 
    'init_mem_cpu_alloc_delta',
    'init_mem_cpu_peaked_delta',
    'init_mem_gpu_alloc_delta',
    'init_mem_gpu_peaked_delta',
    'train_mem_cpu_alloc_delta',
    'train_mem_cpu_peaked_delta',
    'train_mem_gpu_alloc_delta',
    'train_mem_gpu_peaked_delta',
    'acceleration_framework_config_file',
    'output_dir',
    #'error_messages'
]

df, constants = gather_report('benchmark_outputs_final2', raw=False)
df = df[df.columns[~df.columns.isin(remove_columns)]]
df['error_messages'] = df['error_messages'].isna()

def reassemble(df, constants):
    if len(constants) == 0:
        return df
    df = df.copy()
    for k,v in constants.items():
        df[k] = v
    return df


df = reassemble(df, {k:v for k,v in constants.items() if k in ['peft_method']})


def create_dropdown(column_name):
    values = sorted(df[column_name].unique().tolist())
    return gr.Dropdown(
        values, 
        multiselect=True, 
        label=column_name,
        value=values,
    )
    
def select_data(df, column_name, values):
    if len(values) == 0:
        return df
    return df.query(f'{column_name} in @values')

dataframe = gr.Dataframe(
    label="Benchmark Results", 
    #value=df,
    interactive=False
)

train_toks_barplot = gr.BarPlot(
    y="train_tokens_per_second",
    x="throughput",
    color="framework_config",
    group="num_gpus",
    title="train_tokens_per_second",
    group_title="num_gpus",
    tooltip=["model_name_or_path", "framework_config", "peft_method", "train_tokens_per_second"],
    vertical=False,
)

def update(
    model_name_or_path, 
    num_gpus,
    peft_method,
    framework_config,
):
    _df = df
    _df = select_data(_df, 'model_name_or_path', model_name_or_path)
    _df = select_data(_df, 'num_gpus', num_gpus)
    _df = select_data(_df, 'peft_method', peft_method)
    _df = select_data(_df, 'framework_config', framework_config)

    numbers = _df[['model_name_or_path', 'num_gpus', 'framework_config', 'peft_method', 'train_tokens_per_second', 'train_loss'] + list(x for x in df.columns if 'mem' in x)]
    numbers = numbers.drop_duplicates(['model_name_or_path', 'num_gpus', 'framework_config', 'peft_method'])
    TPS = []
    for _, A in numbers.groupby('num_gpus'):
        A['throughput'] = A['train_tokens_per_second'].rank(method='first', ascending=False)
        TPS.append(A)
        
    return numbers, pd.concat(TPS)

demo = gr.Interface(
    fn=update,
    inputs=[
        create_dropdown('model_name_or_path'),
        create_dropdown('num_gpus'),
        create_dropdown('peft_method'),
        create_dropdown('framework_config'),
    ],
    outputs=[
        dataframe, train_toks_barplot
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name='localhost', 
        server_port=7860
    )