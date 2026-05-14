import sys
sys.path.append('./src/')
sys.path.append('../src/')
import NDimModel

import argparse


model = NDimModel.Model()


# loading from csv files
#
# --- simple 2D case (no fractionation) ---
#   python run_batch.py ../input/generic2D_data.csv ../input/generic2D_sources_delta.csv
#
# --- 2D case with Rayleigh-type fractionation, default uniform prior for r ---
#   python run_batch.py ../input/fractionation2D_data.csv ../input/generic2D_sources_delta.csv \
#       --aux_file ../input/fractionation2D_frac.csv
#
# --- 2D fractionation with log-uniform prior for r (useful when r spans decades) ---
#   python run_batch.py ../input/fractionation2D_data.csv ../input/generic2D_sources_delta.csv \
#       --aux_file ../input/fractionation2D_frac.csv \
#       --aux_prior_type loguniform --aux_r_min 0.001 --aux_r_max 1.0
#
# --- 3D case with fractionation ---
#   python run_batch.py ../input/3D_data.csv ../input/3D_sources.csv \
#       --aux_file ../input/3D_frac.csv


parser = argparse.ArgumentParser(description="Run batch processing with specified input files.")
parser.add_argument("data_file", help="Path to the data file")
parser.add_argument("sources_file", help="Path to the sources file")

parser.add_argument("--aux_file", default="", help="Path to optional aux file")
parser.add_argument("--output_dir", default="output", help="Directory to save output files")
parser.add_argument("--output_filenames", default="", help="Custom tag added to output filenames")
parser.add_argument("--output_formats", default="pdf,png", help="Format of output plots (comma-separated list)")
parser.add_argument("--niter", default=1e6, help="Number of max MCMC iterations (terminated when exceeded)")
parser.add_argument("--burnout", default=100, help="Number of burnout iterations (for the likelihood to reach a plateau)")
parser.add_argument("--chain_length", default=500, help="Number of chain entries to be saved")
parser.add_argument("--plot_online", default="True",
                    help="Turn on/off online plotting (True/False, default: True)")

# --- Auxiliary variable prior ---
# These settings apply to the first auxiliary variable 'r'.
# For models with multiple aux variables, call model.set_aux_prior() explicitly
# after model.load_from_fielnames() (see docstring of Model.set_aux_prior).
parser.add_argument(
    "--aux_prior_type",
    default="uniform",
    choices=["uniform", "loguniform"],
    help=(
        "Sampling prior for the auxiliary variable r. "
        "'uniform' draws r ~ Uniform[r_min, r_max] (default). "
        "'loguniform' draws log(r) ~ Uniform[log(r_min), log(r_max)]; "
        "useful when r spans multiple orders of magnitude (requires r_min > 0)."
    ),
)
parser.add_argument("--aux_r_min", default=0.0, type=float,
                    help="Lower bound for the aux variable prior (default: 0.0)")
parser.add_argument("--aux_r_max", default=1.0, type=float,
                    help="Upper bound for the aux variable prior (default: 1.0)")

args = parser.parse_args()

model.load_from_fielnames(
        data_file=args.data_file,
        sources_file=args.sources_file,
        aux_file=args.aux_file,
        )

# Configure the prior for the auxiliary variable 'r' when an aux file was supplied.
# The default (uniform on [0, 1]) matches the original model behaviour.
# Example — switch to a log-uniform prior:
#   model.set_aux_prior('r', prior_type='loguniform', r_min=0.001, r_max=1.0)
if args.aux_file and model.aux_toggle:
    # apply to every aux variable discovered in the model
    for var in model.aux_vars:
        model.set_aux_prior(var,
                            prior_type=args.aux_prior_type,
                            r_min=args.aux_r_min,
                            r_max=args.aux_r_max)

# set the output directory and naming
model.set_outdir(out_dir=args.output_dir)
if args.output_filenames != "":
        model.myfilename = args.output_filenames
else:
        model.myfilename = args.data_file + "_" + args.sources_file + "_" + args.aux_file

# enter your desired formats separated with a comma
# supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
model.myfmt = args.output_formats

model.set_iterations(args.niter, args.burnout, args.chain_length)

# Turn online plotting on/off
model.plotting_switch = str(args.plot_online).strip().lower() not in ("false", "0", "no")

# run model
model.run_model()
