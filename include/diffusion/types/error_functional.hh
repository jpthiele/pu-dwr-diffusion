/**
 * @file error_functional.hh
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele, Uwe Koecher                */
/*                          and Marius Paul Bruchhaeuser                      */
/*                                                                            */
/*  This file is part of pu-dwr-diffusion                                     */
/*                                                                            */
/*  pu-dwr-diffusion is free software: you can redistribute it and/or modify  */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  pu-dwr-diffusion is distributed in the hope that it will be useful,       */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with pu-dwr-diffusion. If not, see <http://www.gnu.org/licenses/>.  */

#ifndef __error_functional_hh
#define __error_functional_hh

namespace diffusion {
namespace types {

enum class error_functional : unsigned int {
    forbidden     = 0,

    /**
     * \f$ J(u) = \frac{1}{| \Omega \times I |_{d+1}}
     * \int_I \int_\Omega u(x,t) \operatorname{d} x \operatorname{d} t \f$.
     */
    mean_global   = 1,

    /**
     * \f$ J(\varphi) = \displaystyle
     * \frac{(\varphi, \hat e)_{\Omega \times I}}
     * {\| \hat e \|_{L^2(I;L^2(\Omega))}} \f$
     * with \f$ \hat e \f$ being a sufficiently good approximation of
     * \f$ e = (u - u_{\tau,h}) \f$.
     */
    L2_L2_global  = 2
};

}}

#endif
