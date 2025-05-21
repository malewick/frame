import sys
sys.path.append('./src/')
sys.path.append('../src/')
import NDimModel

import argparse


model = NDimModel.Model()


# loading from xml file
# model.load_from_xml("../input/3D.xml")

# loading from csv files

# --- simple 2D case ---
#data_file="../input/generic2D_data.csv"
#sources_file="../input/generic2D_sources_delta.csv"
#aux_par_file=""

# --- simple 2D case, many entries ---
#data_file="../input/2D_data.csv"
#sources_file="../input/generic2D_sources_delta.csv"
#aux_par_file=""

# --- 2D case with many indistinguishable sources ---
#data_file="../input/generic2D_data.csv"
#sources_file="../input/many2D_sources.csv"
#aux_par_file=""

# --- 2D case with measurement on the edge ---
#data_file="../input/edge2D_data.csv"
#sources_file="../input/generic2D_sources_delta.csv"
#aux_par_file=""

# --- 2D case with fractionation ---
#data_file="../input/fractionation2D_data.csv"
#sources_file="../input/generic2D_sources_delta.csv"
#aux_par_file="../input/fractionation2D_frac.csv"

# --- 3D case with fractionation ---
#data_file="../input/3D_data.csv"
#sources_file="../input/3D_sources.csv"
#aux_par_file="../input/3D_frac.csv"


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
parser.add_argument("--plot_online", default=True, help="Turn on/off online plotting (True/False)")

# Add additional optional arguments here as needed
args = parser.parse_args()

model.load_from_fielnames(
        data_file=args.data_file,
        sources_file=args.sources_file,
        aux_file=args.aux_file,
        )

# set the output directory and naming
model.set_outdir(out_dir=args.output_dir)
if args.output_filenames=="":
        model.myfilename=args.data_file+"_"+args.sources_file+"_"+args.aux_file

# enter your desired formats separated with a comma
# supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
model.myfmt=args.output_formats

model.set_iterations(args.niter, args.burnout, args.chain_length)

# Trun online plotting on/off
model.plotting_switch=True

# run model
model.run_model()
