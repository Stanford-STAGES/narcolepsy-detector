import os
import tempfile
from time import sleep

# fmt: off
JOBS = [
    # ('r01_prepdat', 'mignot,owners,normal', '05:00:00', '10', 'python prepare_data.py -r 1'),
    # ('r03_prepdat', 'mignot,owners,normal', '03:00:00', '10', 'python prepare_data.py -r 3'),
    # ('r05_prepdat', 'mignot,owners,normal', '02:00:00', '10', 'python prepare_data.py -r 5'),
    # ('r10_prepdat', 'mignot,owners,normal', '01:00:00', '10', 'python prepare_data.py -r 10'),
    # ('r15_prepdat', 'mignot,owners,normal', '01:00:00', '10', 'python prepare_data.py -r 15'),
    # ('r30_prepdat', 'mignot,owners,normal', '01:00:00', '10', 'python prepare_data.py -r 30')
    # ('r0001_prepdat', 'mignot,owners,normal', '05:00:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 1'),
    # ('r0003_prepdat', 'mignot,owners,normal', '05:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 3'),
    # ('r0005_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 5'),
    # ('r0010_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 10'),
    # ('r0015_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 15'),
    # ('r0030_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 30'),
    # ('r0060_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 60'),
    # ('r0150_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 150'),  # 2.5 min
    # ('r0300_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 300'),  # 5 min
    # ('r0600_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 600'),  # 10 min
    # ('r0900_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 900'),  # 15 min
    # ('r1800_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 1800'),  # 30 min
    # ('r2700_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 2700'),  # 45 min
    # ('r3600_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 3600'),  # 60 min
    # ('r5400_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 5400'),  # 90 min
    # ('r7200_prepdat', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 7200'),  # 120 min
    # ('allres_prepdat', 'mignot,owners,normal', '2-00:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -r 1 3 5 10 15 30 60 150 300 600 900 1800 2700 3600 5400 7200 --test')
    # ('r0001_prepdat_ahc', 'mignot,owners,normal', '05:00:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 1'),
    # ('r0003_prepdat_ahc', 'mignot,owners,normal', '05:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 3'),
    # ('r0005_prepdat_ahc', 'mignot,owners,normal', '01:00:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 5'),
    # ('r0010_prepdat_ahc', 'mignot,owners,normal', '01:00:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 10'),
    # ('r0015_prepdat_ahc', 'mignot,owners,normal', '00:30:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 15'),
    # ('r0030_prepdat_ahc', 'mignot,owners,normal', '00:30:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 30'),
    # ('r0060_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 60'),
    # ('r0150_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 150'),  # 2.5 min
    # ('r0300_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 300'),  # 5 min
    # ('r0600_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 600'),  # 10 min
    # ('r0900_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 900'),  # 15 min
    # ('r1800_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 1800'),  # 30 min
    # ('r2700_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 2700'),  # 45 min
    # ('r3600_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 3600'),  # 60 min
    # ('r5400_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 5400'),  # 90 min
    # ('r7200_prepdat_ahc', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/massc/avg_kw21_long -o data/narco_features/avg_kw21_long_ahc -r 7200'),  # 120 min
    # ('r0001_prepdat_usleep', 'mignot,owners,normal', '05:00:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 1'),
    # ('r0003_prepdat_usleep', 'mignot,owners,normal', '05:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 3'),
    # ('r0005_prepdat_usleep', 'mignot,owners,normal', '01:00:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 5'),
    # ('r0010_prepdat_usleep', 'mignot,owners,normal', '01:00:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 10'),
    # ('r0015_prepdat_usleep', 'mignot,owners,normal', '00:30:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 15'),
    # ('r0030_prepdat_usleep', 'mignot,owners,normal', '00:30:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 30'),
    ('r0060_prepdat_usleep', 'mignot,owners,normal', '01:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 60'),
    ('r0150_prepdat_usleep', 'mignot,owners,normal', '01:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 150'),  # 2.5 min
    # ('r0300_prepdat_usleep', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 300'),  # 5 min
    # ('r0600_prepdat_usleep', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 600'),  # 10 min
    # ('r0900_prepdat_usleep', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 900'),  # 15 min
    # ('r1800_prepdat_usleep', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 1800'),  # 30 min
    # ('r2700_prepdat_usleep', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 2700'),  # 45 min
    # ('r3600_prepdat_usleep', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 3600'),  # 60 min
    # ('r5400_prepdat_usleep', 'mignot,owners,normal', '00:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 5400'),  # 90 min
    ('r7200_prepdat_usleep', 'mignot,owners,normal', '01:20:00', '10', 'python prepare_data.py -d data/usleep -o data/narco_features/usleep -r 7200'),  # 120 min
]
# fmt: on


def submit_job(jobname, partition, time, ncpus, command, *args):

    content = f"""#!/bin/bash
#
#SBATCH --job-name={jobname}
#SBATCH -p {partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={ncpus}
#SBATCH --output=/home/groups/mignot/narcolepsy-detector/logs/{jobname}.out
#SBATCH --error=/home/groups/mignot/narcolepsy-detector/logs/{jobname}.err
##################################################

cd $GROUP_HOME/narcolepsy-detector
. activate_env.sh

{command}
"""

    #     print('')
    #     print('#######################################################################################')
    #     print(content)
    #     print('#######################################################################################')
    #     print('')
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.encode())
    os.system(f"sbatch {j.name}")


if __name__ == "__main__":

    print(f"Submitting {len(JOBS)} job(s) ... ")
    for jobinfo in JOBS:
        submit_job(*jobinfo)

    print("All jobs have been submitted!")
