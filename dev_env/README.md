## Development Environment
The experimentation phase was carried out within two distinct environments: Linux Mint and Windows 11. Linux Mint was utilized as the client-side environment, while Windows 11 served as the server-side environment. The hardware specifications for the server-side setup were as follows:

- Intel(R) Core(TM) i5-4670K CPU @ 3.40 GHz
- 32 GB RAM
- 11 GB NVIDIA GeForce GTX 1080 Ti

Python modules for evaluation, training, and visualization were implemented using PyCharm, whereas Jupyter Notebook facilitated the execution of evaluation and training processes.\
Convenient shortcuts in the form of `.bat` files were employed to streamline the initiation of the development environment. The `StartDevelopmentServers.bat` file served as an entry point, orchestrating the execution of `StartDocker.bat`, `StartJN.bat`, and `StartTensorboard.bat`.
The specific purposes of these `.bat` files are outlined as follows:
- `StartDocker.bat`: Initially intended to containerize the development environment via Docker, this approach was abandoned due to challenges in maintaining and updating pip packages within immutable Docker images. Furthermore, Docker offered an indirect advantage. **Although PyCharm's remote development lacked support for Windows server-side setups, Docker provided a feasible workaround. It allowed for the establishment of a connection between the Windows-based server environment and PyCharm for remote development purposes.** Consequently, the `StartDocker.bat` script was repurposed to facilitate the initiation of Docker containers. **To enable remote development in PyCharm using the Docker environment, it's necessary to update the files under the Dockerfile directory.**  
  

- `StartJN.bat`: This script was designed to launch Jupyter Notebook within the designated `notebooks` folder. The notebook interface was made accessible over the LAN through the address 
  

- `StartTensorboard.bat`:  Executing this script triggered the initiation of TensorBoard within the `models` folder. By accessing `localhost:6006/` over the LAN, users could interact with TensorBoard and explore generated graphs.


These streamlined scripts and setups collectively formed the backbone of the development environment, enabling efficient experimentation and analysis.
