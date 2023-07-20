# Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher
# 
# This file is part of pu-dwr-diffusion.
# 
# pu-dwr-diffusion is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# 
# pu-dwr-diffusion is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with pu-dwr-diffusion. If not, see <http://www.gnu.org/licenses/>.


## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2014 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------


MESSAGE("================================================================================")
MESSAGE("=== Configuring DTM++.Project/heat =============================================")
MESSAGE("================================================================================")

# SETs the NAME of the TARGET and PROJECT
SET(TARGET "pu-dwr-diffusion")

FILE(GLOB_RECURSE TARGET_SRC "source/*.cc")
INCLUDE_DIRECTORIES(include)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.0 QUIET HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
IF(NOT ${deal.II_FOUND})
	MESSAGE(FATAL_ERROR "\n*** ERROR: Could not locate a suitably recent version of deal.II. ***\n"
		"SOLUTION A: EITHER you pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake.\n"
		"SOLUTION B: OR, set an environment variable \"DEAL_II_DIR\".\n"
	)
ENDIF()

MESSAGE(STATUS "Found deal.II version ${DEAL_II_PACKAGE_VERSION} at '${deal.II_DIR}'")

SET(_DEALII_GOOD ON)

# IF(NOT DEAL_II_WITH_P4EST)
# 	MESSAGE(SEND_ERROR "\n-- deal.II was build without support for p4est!\n")
# 	SET(_DEALII_GOOD OFF)
# ENDIF()

# IF(NOT DEAL_II_WITH_TRILINOS)
# 	MESSAGE(SEND_ERROR "\n-- deal.II was build without support for Trilinos!\n")
# 	SET(_DEALII_GOOD OFF)
# ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})

MESSAGE("================================================================================")

INCLUDE(cmake/InvokeAutopilot.cmake)
INCLUDE(cmake/AddAdditionalCompilerWarnings.cmake)
INVOKE_AUTOPILOT()

MESSAGE(STATUS "NOTE: Writing configuration into ./detailed.log")
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

INCLUDE(cmake/CreateDetailedLog.cmake)
