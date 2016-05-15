/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file device_free.inl
 *  \brief Inline file for device_free.h.
 */

#include <thrust/detail/config.h>
#include <thrust/device_free.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/detail/malloc_and_free_adl_helper.h>

namespace thrust
{

void device_free(thrust::device_ptr<void> ptr)
{
  using thrust::system::detail::generic::select_system;
  using thrust::system::detail::generic::free;

  typedef thrust::iterator_system< thrust::device_ptr<void> >::type system;

  free(select_system(system()), ptr);
} // end device_free()

} // end thrust
