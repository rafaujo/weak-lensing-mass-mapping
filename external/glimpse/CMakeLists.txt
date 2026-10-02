cmake_minimum_required(VERSION 2.8)

# Paths to external libraries
set(NICAEA_URL https://github.com/EiffL/nicaea.git)

set(SPARSE2D_URL https://github.com/CosmoStat/Sparse2D.git)

set(NFFT_URL https://www-user.tu-chemnitz.de/~potts/nfft/download/nfft-3.3.0.tar.gz)
set(NFFT_MD5 8efe6a43a9565f61ebf836a314b9aeb2)

# Adding customized cmake module
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

include(FindPkgConfig)
include(ExternalProject)

project(glimpse)

# Include external libraries
find_package(GSL    REQUIRED)
find_package(CCFits REQUIRED)
find_package(FFTW   REQUIRED)
find_package(NFFT   REQUIRED)
find_package(Boost  COMPONENTS program_options REQUIRED)

# Include external projects
## NICAEA
ExternalProject_Add(NICAEA
        PREFIX NICAEA
        GIT_REPOSITORY ${NICAEA_URL}
        GIT_TAG f7850e5009d0341226d4c1f38b116f316d8a0516
        CMAKE_ARGS
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=-fcommon
            -DCMAKE_CXX_STANDARD=14
        BUILD_COMMAND
            ${CMAKE_COMMAND} --build <BINARY_DIR> --target nicaea
        INSTALL_COMMAND "")
        set(NICAEA_LIBRARY_DIR ${CMAKE_BINARY_DIR}/NICAEA/src/NICAEA/lib)
        set(NICAEA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/NICAEA/src/NICAEA/include)
        set(NICAEA_LIBRARIES -lnicaea)

## Sparse2D
ExternalProject_Add(SPARSE2D
        PREFIX SPARSE2D
        GIT_REPOSITORY ${SPARSE2D_URL}
        GIT_TAG v2.1.2
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extern
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_CXX_STANDARD=14
            -DSPARSE3D=OFF
            -DBUILD_MSVST=OFF)
        set(SPARSE2D_LIBRARY_DIR ${CMAKE_BINARY_DIR}/extern/lib)
        set(SPARSE2D_INCLUDE_DIR ${CMAKE_BINARY_DIR}/extern/include)
        set(SPARSE2D_LIBRARIES -lsparse2d -lsparse1d -ltools)
        
include_directories(${SPARSE2D_INCLUDE_DIR}
		    ${GSL_INCLUDE_DIR}
		    ${CCFITS_INCLUDE_DIR}
		    ${CFITSIO_INCLUDE_DIR}
		    ${Boost_INCLUDE_DIR}
		    ${NICAEA_INCLUDE_DIR}
		    ${NFFT_INCLUDE_DIR}
		    ${FFTW_INCLUDE_DIR})
		    
link_directories(${SPARSE2D_LIBRARY_DIR}
		    ${GSL_LIBRARY_DIR}
		    ${CCFITS_LIBRARY_DIR}
		    ${CFITSIO_LIBRARY_DIR}
		    ${Boost_LIBRARY_DIR}
		    ${NICAEA_LIBRARY_DIR}
		    ${NFFT_LIBRARY_DIR}
		    ${FFTW_LIBRARY_DIR})

set(GLIMPSE_SRC src/survey.cpp
		src/redshift_distribution.cpp
		src/field.cpp
		src/surface_reconstruction.cpp
		src/starlet_2d.cpp
		src/wavelet_transform.cpp)

if(APPLE)
  set(CMAKE_CXX_FLAGS "-DMACOS -O3 -fpermissive")
  set(CMAKE_C_FLAGS "-O3 -fpermissive")
else(APPLE)
  set(CMAKE_CXX_FLAGS "-O3 -fpermissive -fopenmp")
  set(CMAKE_C_FLAGS "-O3 -fpermissive")
endif(APPLE)
  
add_executable(glimpse src/glimpse.cpp ${GLIMPSE_SRC})
add_dependencies(glimpse NICAEA SPARSE2D)
target_link_libraries(glimpse ${NICAEA_LIBRARIES}
			      ${SPARSE2D_LIBRARIES}
			      ${Boost_LIBRARIES} 
			      ${GSL_LIBRARIES}
			      ${NFFT_LIBRARIES}
			      ${FFTW_LIBRARIES}
			      ${CCFITS_LIBRARY}
			      ${CFITSIO_LIBRARY})
