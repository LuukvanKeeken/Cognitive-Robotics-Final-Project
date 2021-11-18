cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(GOOD_Descriptor)
add_executable(goodexe main.cpp)

find_package(PCL 1.7 REQUIRED)

if (PCL_FOUND)
else (PCL_FOUND)
	MESSAGE("PCL 1.7 not found. You must execute the following command\nsudo apt-get install libpcl-1.7-all-dev\n")
endif (PCL_FOUND)

include_directories( ${PCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

add_library (good good.cpp)
SET_TARGET_PROPERTIES(good PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(good ${PCL_LIBRARIES})
target_link_libraries(good boost_system)

target_link_libraries(goodexe ${PCL_LIBRARIES} )
target_link_libraries(goodexe boost_system)
target_link_libraries(goodexe good)