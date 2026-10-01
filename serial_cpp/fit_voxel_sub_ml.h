// Parameter fitting for the sub-diffusion model using the Mittag-Leffler function.

FitResult fit_voxel_sub_ml(const std::array<double, NB>& Svec,);

struct FitResult
{
    double D;
    double beta;
    double err;
};