-----------------------------------------------------------------------------------------------

FEmodels: a Python 3 package for nonliner panel and network formation models with fixed effects 

-----------------------------------------------------------------------------------------------

----------------------
Developer & maintainer
----------------------
Zizhong Yan, IESR of Jinan University, e-mail: helloyzz@gmail.com



This package contains three commands:

---------------------
1. `NetworkFE`
Version: v.0.8.1-beta
---------------------
[> Description <] 

Please refer to the help documentation by using the command `help(NetworkFE)`.


[> Dependencies <]
The `NetworkFE` command depends on the `netrics` package (Graham, 2016). Users do not 
need to separately download and install the `netrics` package as it is already included 
in the `./dependency` folder and will be automatically loaded.


[> References <]
Graham, Bryan S. (2016). "netrics: a Python 3.7  package for econometric analysis of 
	networks," (Version 0.0.1) [Computer program]. Available at 
	https://github.com/bryangraham/netrics (Accessed 04 October 2018)
Graham, Bryan S. (2017). "An econometric model of link formation with degree 
	heterogeneity," Econometrica 85 (4): 1033 - 1063


[> Authors of the command <]
Zizhong Yan (helloyzz@gmail.com)
Shiran Hu (, contribute to extending of additional functionality to undirected network case, 
             and condition logit in directed network setting)



---------------------
2. `PanelFE`
Version: v.0.8.1-beta
---------------------
[> Description <] 

Please refer to the help documentation by using the command `help(PanelFE)`.



[> Dependencies <]
The `PanelFE` command depends on the PyTorch package to perform autograd. Please note 
that PyTorch is not included in the standard Anaconda distribution, so users may need to 
install the PyTorch package. See pytorch.org.



[> References <]



[> Author of the command <]
Zizhong Yan (helloyzz@gmail.com)



---------------------
3. `CoevoFE`
Version: v.0.5.1-beta
---------------------
[> Description <] 

Please refer to the help documentation by using the command `help(CoevoFE)`.

Author of CoevoFE: Zizhong Yan


[> C++ acceleration <]
To accelerate the `CoevoFE` algorithm, several intensive procedures have been recoded 
using Cython. The compiled C++ codes provided with the package support macOS platform 
for Python 3.8, 3.9, and 3.12, as well as Windows platform for Python 3.12. 

To enable the acceleration, users may install Cython via 'pip install cython' or 'conda
install cython' methods. Please see cython.org for further information.

To build C++ code for other versions of Python, users should install the GNU Compiler 
Collection (GCC) for C++ on macOS or Microsoft C++ Build Tools on the Windows platform. 
For more information, visit https://cython.readthedocs.io/en/latest/src/quickstart/build.html.


[> References <]



[> Author of the command <]
Zizhong Yan (helloyzz@gmail.com)




------------------------
COMPATIBILITY & SUPPORTS
------------------------
This package is developed using Python 3.9 and has been tested on Python 3.8 to 3.12 
on both macOS and Windows platforms.

This package is offered "as is", without warranty, implicit or otherwise. While I would
appreciate bug reports, suggestions for improvements and so on, I am unable to provide any
meaningful user-support. Please e-mail Zizhong Yan at helloyzz@gmail.com.


------------------
PACKAGE CITATION
------------------
Yan, Zizhong and Shiran Hu. (2024). "FEmodels: a Python 3 package for nonline panel data, network 
	formation models with fixed effects ," (Version 0.0.1) [Computer program]. Available at 
	https://github.com/zizhongyan/FEmodels (Accessed 30 October 2024) 

Please cite both the package and the underlying source articles listed above when using this 
code in your research.








































