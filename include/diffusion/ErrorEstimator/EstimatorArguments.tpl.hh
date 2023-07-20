/**
 * @file EstimatorArguments.tpl.hh
 *
 * @author Jan Philipp Thiele (JPT)
 */

/*  Copyright (C) 2012-2023 by Jan Philipp Thiele                             */
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

#ifndef __EstimatorArguments_tpl_hh
#define __EstimatorArguments_tpl_hh

#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <vector>

namespace diffusion {
namespace dwr {

namespace estimator {

class Arguments{
public:
    struct{
        std::shared_ptr<dealii::Vector<double>> u_kh; //low order dG0

        std::shared_ptr<dealii::Vector<double>> z_k;  //high order dG0
    } tm; //interval start

    struct{
        std::shared_ptr<dealii::Vector<double>> u_k;  //high order dG0
        std::shared_ptr<dealii::Vector<double>> u_kh; //low order dG0

        std::shared_ptr<dealii::Vector<double>> z_k;  //high order dG0
        std::shared_ptr<dealii::Vector<double>> z_kh; //low order dG0
    } tn; //interval end

    struct{
        std::shared_ptr<dealii::Vector<double>> z_kh; //low order dG0

        std::shared_ptr<dealii::Vector<double>> u_k;  //high order dG0
    } tnp1;
};


}}}//namespaces


#endif
