# PyGLDA: a fine-scale Python-based Global Land Data Assimilation system for integrating satellite gravity data into hydrological models

#### Description
Data assimilation (DA) has been widely used to merge the hydrological 'model' and the remote sensing 'observations', to complement each other towards a better temporal, horizontal and vertical disaggregation of the TWS estimates.
In this study, we present a free, open-source, parallel and Python based PyGDA system to address the challenges of the High-Resolution Global DA (HRGDA) for GRACE and W3RA hydrologocal model, where the framework has been completely re-designed relative to previous studies to be able to run HRGDA (up to 0.1 degree and daily) efficiently and stably.

<img src="structure.jpg" width="450">

#### Contact
Fan Yang (fany@plan.aau.dk) , Ehsan Forootan (efo@plan.aau.dk), Maike Schumacher (maikes@plan.aau.dk)

Geodesy Group (see https://aaugeodesy.com/), Department of Sustainability and Planning, Aalborg University, Aalborg 9000, Denmark

This work is supported by the Danmarks Frie Forskningsfond [10.46540/2035-00247B] through the DANSk-LSM project. Additional supports come from and national Natural Science Foundation of China (Grant No. 42274112 and No. 41804016). We also acknowledge the support of W3RA model via https://www.dropbox.com/scl/fo/b0hneugr9vao0rqm4oh86/AEPPU-QG6kgh9wTlIBgiwMQ?rlkey=q7ux08mitdghnoac3e4spwaev&e=1&dl=0


#### Features
1. A novel unified framework to enable it to address both the basin-scale and grid-scale DA, where the area to be computed is defined as the 'basin' and the grid is treated as the 'sub-basin'. Any shape file that has well defined the basin and sub-basins is accepted to launch PyGDA;
2. A novel framework to seamlessly transit from regional DA to global DA by introducing domain localization and weighting algorithms; 
3. Computation of the spatial covariance of the observations, i.e., GRACE based TWS, is available between arbitrary sub-basins/grids for the first time. Accounting for the spatial covariance into the DA is also allowed in PyGDA;
4. Flexible options in the grid resolution: a choice of 0.1 degree and 0.5 degree is available for the employed global W3RA model; a flexible choice of the grid/sub-basin to be assimilated, e.g., from 1 degree to 5 degree with an increment of 0.5 degree;
5. Flexible choice of perturbation, e.g., which forcing data and which model parameters to be perturbed in which noise distribution (Gaussian or Triangle; additive or multiplicative);
6. In addition to the general Ensemble Kalman filter, a novel Kalman smoother that achieves optimal temporal disaggregation from monthly increment to daily increment is facilitated.

#### Software Architecture
1. A flexible modular structure to de-couple PyGDA into three individual modules: (1) hydrological model, (2) GRACE processing and (3) mathematical DA integration. This makes possible to easily develop/modify/replace any individual module;
2. A high-level programming language (object-oriented Python) for easy comprehension of the code and to facilitate extensibility. The Python translation of W3RA model is distributed;
3. Easy/fast installation for cross-platform to the users’ needs and capacity (e.g., Windows, Linux and parallel computation at high performance clusters);
4. Optimization by using high-performance Numpy package as the basic data container to reach comparable numerical efficiency as C++, Fortran and Matlab;
5. User-friendly interaction by (1) controlling/configuring PyGDA with JSON setting file, where a wide options are available for a different purpose; and (2) state-of-the-art data structure H5DF for reading and writing spatiotemporal data to allow for efficient management of data storage.

#### Installation
Please follow below the step-by-step instruction to do the installation, given that the Conda environment has been established already. A simpler installation via `.yml` to copy the environment will also be released soon. While PyGLDA does not specify the version of Python, it is strongly recommended to use python > 3.9. 
1.  pip install netcdf4
2.  conda install metview-batch  -c conda-forge
3.  conda install metview-python  -c conda-forge
4.  pip install cdsapi
5.  conda install -c conda-forge mpi4py mpich
6.  pip install h5py
7.  pip install scipy
8.  pip install geopandas

To test if it is properly installed, please type below command and see if it passes.
1. mpiexec -n 5 python -m mpi4py.bench helloworld
2. metview -slog
3. python3 -m metview selfcheck

(Optionally) To make the visualization function, one has to install PyGMT (better to create an individual environment for its installation), see https://www.pygmt.org/latest/.

Potential installation troubles:
1. metview-related issues please refer to https://metview.readthedocs.io/en/latest/index.html
2. mpi4py-related issues please refer to https://mpi4py.readthedocs.io/en/latest/install.html


#### Instructions
Three demo (demo1, demo2, demo3) are present under `/py-w3ra/demo` to showcase the use of PyGLDA. Each demo has its detailed comments in its script. To run the demo, a bunch of sample data are necessary to be installed.
The sample data is distributed together with the code at the given data repository, named after `External Data`.
To begin with the demo, we suggest to place the sample data at its default place, which is under `/PyGLDA/External Dara/`.
Nevertheless, as an advanced user, one can place data at any desired place as long as the setting files are well configured.
Below we give a brief introduction of three demo.
1.  demo_1.py.  In this demo, we show how to update the meteorological forcing field (ERA5-land) online
2.  demo_2.py.  In this demo, we show how to configure and implement the W3RA water balance model.
3.  demo_3.py.  In this demo, we show how to implement basin-scale regional data assimilation, grid-scale data assimilation and 
the grid-scale global data assimilation.
Please feel free to contact us for possible more examples. 

