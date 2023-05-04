import asyncio
import argparse
import os

from trainml.trainml import TrainML

parser = argparse.ArgumentParser(
    description="Instruction-Trained Large Language Model (LLM) Endpoint Example"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The language model checkpoint to use",
)
parser.add_argument(
    "--gpu-count",
    type=int,
    default=2,
    help="Number of GPUs to attach (max 4)",
)


async def create_endpoint(trainml, checkpoint, gpu_count):
    job = await trainml.jobs.create(
        "Instruction-Trained LLM Endpoint Example",
        type="endpoint",
        gpu_types=["rtx3090"],
        gpu_count=gpu_count,
        disk_size=10,
        endpoint=dict(
            routes=[
                dict(
                    path=f"/instruct",
                    verb="POST",
                    function="instruct",
                    file="inference",
                    body=[
                        dict(
                            name="instruction",
                            type="str",
                            positional=True,
                        ),
                        dict(
                            name="input",
                            type="str",
                            optional=True,
                            positional=False,
                        ),
                        dict(
                            name="max_tokens",
                            type="int",
                            optional=True,
                            positional=False,
                        ),
                        dict(
                            name="temperature",
                            type="float",
                            optional=True,
                            positional=False,
                        ),
                        dict(
                            name="top_p",
                            type="float",
                            optional=True,
                            positional=False,
                        ),
                        dict(
                            name="num_beams",
                            type="int",
                            optional=True,
                            positional=False,
                        ),
                    ],
                ),
            ]
        ),
        model=dict(
            source_type="local",
            source_uri=os.getcwd(),
            checkpoints=[checkpoint],
        ),
        environment=dict(
            type="DEEPLEARNING_PY39",
        ),
    )
    return job


if __name__ == "__main__":
    args = parser.parse_args()
    trainml = TrainML()
    job = asyncio.run(
        create_endpoint(
            trainml,
            args.checkpoint,
            args.gpu_count,
        )
    )
    print("Created Endpoint: ", job.id, " Waiting to Start...")
    asyncio.run(job.connect())
    asyncio.run(job.wait_for("running"))
    print("Job ID: ", job.id, " Running")
    asyncio.run(job.disconnect())
    print("URL", job.url)
