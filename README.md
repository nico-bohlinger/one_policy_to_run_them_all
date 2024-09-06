# One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion

This repository provides the implementation for the paper:

<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/one_policy_to_run_them_all.pdf" target="_blank">
      <b> One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion </b>
      </a>
      <br>
      Nico Bohlinger, Grzegorz Czechmanowski, Maciej Krupka, Piotr Kicki, Krzysztof Walas, Jan Peters and Davide Tateo
      <br>
      <em>Conference on Robot Learning</em>, 2024
      <br>
      <a href="https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/one_policy_to_run_them_all.pdf">paper</a> /
      <a href="https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/one_policy_to_run_them_all.pdf" target="_blank">project page</a>
    <br>
</td>

<br>
<img src="image.png"/>

## Installation
1. Install RL-X

Default installation for a Linux system with a NVIDIA GPU.
For other configurations, see the RL-X [documentation](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide).
```bash
conda create -n one_policy_to_run_them_all python=3.11.4
conda activate one_policy_to_run_them_all
git clone git@github.com:nico-bohlinger/RL-X.git
cd RL-X
pip install -e .[all] --config-settings editable_mode=compat
pip uninstall $(pip freeze | grep -i '\-cu12' | cut -d '=' -f 1) -y
pip install "torch>=2.2.1" --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

2. Install the project
```bash
git clone git@github.com:nico-bohlinger/one_policy_to_run_them_all.git
cd one_policy_to_run_them_all
pip install -e .
```

**Note**: If package versions are conflicting, due to newer versions of RL-X, modify RL-X's dependencies to align with the requirements.txt file in this repository as it contains the exact package versions used for this project.


## Training a model
1. Run the following commands to start an experiment
```bash
cd one_policy_to_run_them_all/experiments
sbatch experiment.sh
```


## Testing a trained model
1. Move the trained model to the experiments folder
2. Run the following commands to test a trained model
```bash
cd one_policy_to_run_them_all/experiments
sbatch test.sh
```
#### Controlling the robots
Either modify the commands.txt file, where the values are target x, y and yaw velocities, or connect a **Xbox 360** controller and control the target x,y velocity with the left joystick and the yaw velocity with the right joystick.

To switch the robot, either change the robot id in the multi_render.txt file or press the LB and RB button on the controller.

## Citation
If you use or refer to this repository in your research, please cite:

```
@article{bohlinger2024onepolicy,
    title={One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion},
    author={Bohlinger, Nico and Czechmanowski, Grzegorz and Krupka, Maciej and Kicki, Piotr and Walas, Krzysztof and Peters, Jan and Tateo, Davide},
    journal={Conference on Robot Learning},
    year={2024}
}
```

## License
The robot assets and XML files do not belong to the authors of this repository.
They are used for research purposes only and are not covered by the MIT license. The MIT license only covers the code in this repository.