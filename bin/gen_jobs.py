from os import system
from sys import argv

# job params
time = "03:59:00"
job_id = argv[1]  # "0100"
train_with_noise = argv[2]  # 0 or 1
force_continuity = argv[3]  # 0 or 1
pass_dataset_cl = argv[4]  # 0 or 1
pass_dataset_ss = argv[5]  # 0 or 1
pass_dataset_clss = argv[6]  # 0 or 1
arch = f"-a {argv[7]}"  # arch20
epoch = int(argv[8])
trb = int(argv[9])


noise = "n" if train_with_noise == "1" else "0"
continuity = "c" if force_continuity == "1" else "0"
nc = "-nc" if continuity == "0" else ""
job_name = f"{job_id}_{noise}_{continuity}"

# training
dataset_cl_arg = "-dcl kegg_ndx_ACTp" if pass_dataset_cl == "1" else ""
dataset_ss_arg = "-dss pdb_ndx_ss" if pass_dataset_ss == "1" else ""
dataset_clss_arg = "-dclss pdb_act_clss" if pass_dataset_clss == "1" else ""

noise_arg = "-no 0.05" if train_with_noise == "1" else ""
epoch_arg = f"-e {epoch}"
trb_arg = f"-trb {trb}"
system(f"mkdir exp{job_id}_jobs")


for mid_val in range(0, 1000, epoch):
    mid = mid_val if mid_val == 0 else mid_val - 1
    mvid_arg = f"-mvid VERSION#{mid}" if mid > 0 else ""
    next_job = mid + epoch if mid_val != 0 else mid + epoch - 1
    job_script_template = f"""#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time={time}             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                   # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks-per-node=1         # number of CPUs (or cores) per task (same as -c)
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=10G                    # memory required per allocated CPU (or SeqEN) - amount of memory (in bytes)
#SBATCH --job-name={job_name}           # you can give your job a name for easier identification (same as -J)

########## Command Lines for Job Running ##########
conda activate SeqEN

cd /mnt/home/nayebiga/SeqEncoder/SeqEN/
module load GCCcore/11.1.0 Python/3.9.6
module load GCC/11.1.0-cuda-11.4.2  CUDA/11.4.2
python3 ./SeqEN2/sessions/train_session.py -n experiment{job_id} {dataset_cl_arg} {dataset_ss_arg} {dataset_clss_arg} {arch} {noise_arg} {nc} {trb_arg} -w 20 -d1 8 -ts params4 -ti 100 -le 1000 {epoch} {mvid_arg}

scontrol show job $SLURM_JOB_ID                                       ### write job information to SLURM output file.
js -j $SLURM_JOB_ID                                                   ### write resource usage to SLURM output file (powertools command).

cd /mnt/home/nayebiga/SeqEncoder/SeqEN/models/jobs/exp{job_id}_jobs
sbatch {next_job}.sb

"""

    with open(f"./exp{job_id}_jobs/{mid}.sb", "w") as file:
        file.write(job_script_template)
