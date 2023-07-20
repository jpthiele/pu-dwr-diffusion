/**
 * @file ExactSolution_Simple.tpl.cc
 *
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * 
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

#include <diffusion/ExactSolution/ExactSolution_Simple.tpl.hh>

// DEAL.II includes

// C++ includes

namespace diffusion {
namespace exact_solution {

template<int dim>
double
Simple<dim>::
value(
    const dealii::Point<dim> &x,
    [[maybe_unused]]const unsigned int c
) const {
    Assert(c==0, dealii::ExcMessage("you want to get component value which is not implemented"));
    Assert(dim==2, dealii::ExcNotImplemented());

    const double t{this->get_time()};

    const double gx = 0.5*(x[0]*x[0]-x[0]);
    const double gy = 0.5*(x[1]*x[1]-x[1]);

    return -gx*gy*t;
}

template<int dim>
dealii::Tensor<1,dim>
Simple<dim>::
gradient(
    const dealii::Point<dim> &x,
    [[maybe_unused]]const unsigned int c
) const{
    Assert(c==0, dealii::ExcMessage("you want to get component value which is not implemented"));
    Assert(dim==2, dealii::ExcNotImplemented());

    const double t{this->get_time()};

    const double gx = 0.5*(x[0]*x[0]-x[0]);
    const double gy = 0.5*(x[1]*x[1]-x[1]);


    const double gx_prime = x[0]-0.5;
    const double gy_prime = x[1]-0.5;

    dealii::Tensor<1,dim> res;

    res[0] = -gx_prime*gy*t;
    res[1] = -gx*gy_prime*t;

    return res;

}
}} //namespaces

#include "ExactSolution_Simple.inst.in"
