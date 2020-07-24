#echo "[sver.unix_build] checking if build dir should be removed"

#export FAILCMD='{ echo "FAILED VTOOL BUILD" ; exit 1; }'

#python2.7 -c "import utool as ut; print('keeping build dir' if ut.get_argflag('--no-rmbuild') else ut.delete('build'))" $@

#mkdir build
#cd build

#echo "$OSTYPE"

#export PYEXE=$(which python2.7)
#if [[ "$VIRTUAL_ENV" == ""  ]]; then
#    export LOCAL_PREFIX=/usr/local
#    export _SUDO="sudo"
#else
#    export LOCAL_PREFIX=$($PYEXE -c "import sys; print(sys.prefix)")/local
#    export _SUDO=""
#fi

#echo "LOCAL_PREFIX = $LOCAL_PREFIX"

#if [[ "$OSTYPE" == "darwin"* ]]; then
#    cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV ..  || $FAILCMD
#elif [[ "$OSTYPE" == "msys"* ]]; then
#    echo "USE MINGW BUILD INSTEAD" ; exit 1
#    export INSTALL32="c:/Program Files (x86)"
#    echo "INSTALL32=$INSTALL32"
#    cmake -G "MSYS Makefiles" -DOpenCV_DIR="$INSTALL32/OpenCV" .. || $FAILCMD
#else
#    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV ..  ||  $FAILCMD
#fi

#if [[ "$OSTYPE" == "msys"* ]]; then
#    make ||  $FAILCMD
#else
#    export NCPUS=$(grep -c ^processor /proc/cpuinfo)
#    make -j$NCPUS ||  $FAILCMD
#fi
cd src/
make
cp -v flukematch_lib.so ../wbia_flukematch
make -f Makefile_OC_WDTW
cp -v oc_wdtw.so ../wbia_flukematch
cd ../
pip install -e .
#cp -v libsver* ../vtool
#cd ..
