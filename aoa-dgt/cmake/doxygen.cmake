# Helper macro to add a "doc" target with CMake build system.
# and configure doxy.config.in to doxy.config
#
# Please note, that the tools, e.g.:
# doxygen, dot, latex, dvips, makeindex, gswin32, etc.
# must be in path.
#
# adapted from work of Jan Woetzel 2004-2006
# www.mip.informatik.uni-kiel.de/~jw

find_package(Doxygen)

if (DOXYGEN_FOUND)
  if (OPSYS STREQUAL "MACOSX")
    set(GENERATE_DOCSET "YES")
  else (OPSYS STREQUAL "MACOSX")
    set(GENERATE_DOCSET "NO")
  endif (OPSYS STREQUAL "MACOSX")

  set(doxyfile_in ${CMAKE_SOURCE_DIR}/doc/cupoly.doxygen.in)
  set(doxyfile ${CMAKE_BINARY_DIR}/doc/cupoly.doxygen)

	
	if (EXISTS ${doxyfile_in})
		message(STATUS "Configured ${doxyfile_in}")
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/cupoly.doxygen.in
			${CMAKE_CURRENT_BINARY_DIR}/doc/cupoly.doxygen @ONLY )
		# use config from BUILD tree
		set(DOXY_CONFIG ${doxyfile})
	else (EXISTS ${doxyfile_in})
		# use config from SOURCE tree
		if (EXISTS ${doxyfile})
			message(STATUS "Using existing ${doxyfile}")
			set(DOXY_CONFIG ${doxyfile})
		else (exists ${doxyfile})
			# failed completely...
			message(SEND_ERROR "Please create ${doxyfile_in} (or doxy.config as fallback)")
		endif(EXISTS ${doxyfile})

	endif(EXISTS ${doxyfile_in})

	ADD_CUSTOM_TARGET(doc ALL ${DOXYGEN_EXECUTABLE} ${doxyfile} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} COMMENT "Generating API documentation with Doxygen" VERBATIM)

endif(DOXYGEN_FOUND)

