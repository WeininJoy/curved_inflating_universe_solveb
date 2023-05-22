# this code cannot be run directly
# do 'source /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin/clik_profile.csh' from your csh shell or put it in your profile

 

if !($?PATH) then
setenv PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin
else
set newvar=$PATH
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin:@:@g`
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin\$@@` 
set newvar=`echo ${newvar} | sed s@^/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin:@@`  
set newvar=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin:${newvar}                     
setenv PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/bin:${newvar} 
endif
if !($?PYTHONPATH) then
setenv PYTHONPATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages
else
set newvar=$PYTHONPATH
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages:@:@g`
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages\$@@` 
set newvar=`echo ${newvar} | sed s@^/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages:@@`  
set newvar=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages:${newvar}                     
setenv PYTHONPATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib/python/site-packages:${newvar} 
endif
if !($?LD_LIBRARY_PATH) then
setenv LD_LIBRARY_PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1
else
set newvar=$LD_LIBRARY_PATH
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1:@:@g`
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1\$@@` 
set newvar=`echo ${newvar} | sed s@^/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1:@@`  
set newvar=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1:${newvar}                     
setenv LD_LIBRARY_PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1:${newvar} 
endif
if !($?LD_LIBRARY_PATH) then
setenv LD_LIBRARY_PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib
else
set newvar=$LD_LIBRARY_PATH
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib:@:@g`
set newvar=`echo ${newvar} | sed s@:/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib\$@@` 
set newvar=`echo ${newvar} | sed s@^/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib:@@`  
set newvar=/home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib:${newvar}                     
setenv LD_LIBRARY_PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/lib:${newvar} 
endif
setenv CLIK_PATH /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1

setenv CLIK_DATA /home/weinin/Documents/Research/Will_solve_b/Cobaya_new_packages/packages/code/planck/code/plc_3.0/plc-3.1/share/clik

setenv CLIK_PLUGIN rel2015

