/**
 * @file Force_Hartmann142.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele and Uwe Koecher             */
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

#include <diffusion/Force/Force_Hartmann142.tpl.hh>

// DEAL.II includes

// C++ includes

namespace diffusion {
namespace force {

template<int dim>
double
Hartmann142<dim>::
value(
    const dealii::Point<dim> &x,
    [[maybe_unused]]const unsigned int c
) const {
    Assert(c==0, dealii::ExcMessage("you want to get component value which is not implemented"));
    Assert(dim==2, dealii::ExcNotImplemented());
	
    const double t{this->get_time()};
	
    const double x0 = 0.5+0.25*std::cos(2.*M_PI*t);
    const double x1 = 0.5+0.25*std::sin(2.*M_PI*t);
	
    const double denom = 1. + a*( (x[0]-x0)*(x[0]-x0) + (x[1]-x1)*(x[1]-x1) );

    double dtu =
        -( ( a * (x[0]-x0) * M_PI * std::sin(2.*M_PI*t) ) - ( a * (x[1]-x1) * M_PI * std::cos(2.*M_PI*t)) ) /
        (denom*denom);

    const double u_xx =
        -2.*a*( 1./(denom*denom)
            +2.*a*(x[0]-x0)*(x[0]-x0) * (-2./(denom*denom*denom))
        );

    const double u_yy =
        -2.*a*( 1./(denom*denom)
            +2.*a*(x[1]-x1)*(x[1]-x1) * (-2./(denom*denom*denom))
        );

    return dtu - epsilon * (u_xx+u_yy);
}

}} //namespaces

#include "Force_Hartmann142.inst.in"
