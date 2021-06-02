import sys
sys.path.append('./src/')
import NDimModel


model = NDimModel.Model()


# loading from xml file
# model.load_from_xml("input/3D.xml")

# loading from csv files

# --- simple 2D case ---
#data_file="input/generic2D_data.csv"
#sources_file="input/generic2D_sources_delta.csv"
#aux_par_file=""

# --- simple 2D case, many entries ---
data_file="input/2D_data.csv"
sources_file="input/generic2D_sources_delta.csv"
aux_par_file=""

# --- 2D case with many indistinguishable sources ---
#data_file="input/generic2D_data.csv"
#sources_file="input/many2D_sources.csv"
#aux_par_file=""

# --- 2D case with measurement on the edge ---
#data_file="input/edge2D_data.csv"
#sources_file="input/generic2D_sources_delta.csv"
#aux_par_file=""

# --- 2D case with fractionation ---
#data_file="input/generic2D_data.csv"
#sources_file="input/generic2D_sources_delta.csv"
#aux_par_file="input/fractionation2D_frac.csv"

# --- 3D case with fractionation ---
#data_file="input/3D_data.csv"
#sources_file="input/3D_sources.csv"
#aux_par_file="input/3D_frac.csv"


model.load_from_fielnames(
        data_file=data_file,
        sources_file=sources_file,
        aux_file=aux_par_file,
        )


# set the output directory
model.myfilename="2Dmult"
# supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
# enter your desired formats separated with a comma
model.myfmt="pdf,png"
model.set_outdir(out_dir="test_output")


#model.set_iterations(1e6, 1e3, 100)
model.set_iterations(1e5, 100, 700)

# Trun online plotting on/off
model.plotting_switch=True

# run model
model.run_model()
