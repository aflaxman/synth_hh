# Use the bash shell to interpret this job script
#$ -S /bin/bash
#

# submit this job to nodes that have
# at least 8 GB of RAM free.
#$ -l mem_free=8.0G

cd /ihme/homes/abie/projects/2021/synth_hh

## Put the hostname, current directory, and start date
## into variables, then write them to standard output.
GSITSHOST=`/bin/hostname`
GSITSPWD=`/bin/pwd`
GSITSDATE=`/bin/date`
echo "**** JOB STARTED ON $GSITSHOST AT $GSITSDATE"
echo "**** JOB RUNNING IN $GSITSPWD"
##

# make sure that boost library is in the path
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/boost-current/lib

echo calling python "$@"
python "$@"


## Put the current date into a variable and report it before we exit.
GSITSENDDATE=`/bin/date`
echo "**** JOB DONE, EXITING 0 AT $GSITSENDDATE"
##

## Exit with return code 0
exit 0

