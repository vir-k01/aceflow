from jobflow import Flow
from ace_jobflow.flows.data_gen import data_gen_flow
from ace_jobflow.jobs.data import read_outputs
from ace_jobflow.jobs.train import naive_train_ACE


def naive_flow(compositions: list, num_points: int = 5, temperature: float = 2000, max_steps: int = 2000, batch_size: int = 200, gpu_index: int | None = None):
    data = data_gen_flow(compositions, num_points=num_points, temperature=temperature)
    read_job = read_outputs(data.output)
    trainer = naive_flow(read_job.output, max_steps=max_steps, batch_size=batch_size, gpu_index=gpu_index)
    return Flow([data, read_job, trainer], output=trainer.output, name='ACE_wf_Test')