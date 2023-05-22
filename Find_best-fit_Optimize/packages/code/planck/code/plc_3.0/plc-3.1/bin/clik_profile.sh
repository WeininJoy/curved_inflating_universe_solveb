# this code cannot be run directly
# do 'source /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin/clik_profile.sh' from your sh shell or put it in your profile

function addvar () {
local tmp="${!1}" ;
tmp="${tmp//:${2}:/:}" ; tmp="${tmp/#${2}:/}" ; tmp="${tmp/%:${2}/}" ;
export $1="${2}:${tmp}" ;
} 

if [ -z "${PATH}" ]; then 
PATH=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin
export PATH
else
addvar PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin
fi
if [ -z "${PYTHONPATH}" ]; then 
PYTHONPATH=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages
export PYTHONPATH
else
addvar PYTHONPATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages
fi
if [ -z "${LD_LIBRARY_PATH}" ]; then 
LD_LIBRARY_PATH=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1
export LD_LIBRARY_PATH
else
addvar LD_LIBRARY_PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1
fi
if [ -z "${LD_LIBRARY_PATH}" ]; then 
LD_LIBRARY_PATH=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib
export LD_LIBRARY_PATH
else
addvar LD_LIBRARY_PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib
fi
CLIK_PATH=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1
export CLIK_PATH

CLIK_DATA=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/share/clik
export CLIK_DATA

CLIK_PLUGIN=rel2015
export CLIK_PLUGIN

