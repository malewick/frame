import sys
sys.path.append('./src/')
import NDimModel


model = NDimModel.Model()


# loading from xml file
# model.load_from_xml("input/3D.xml")

# loading from csv files

#data_file="input/3D_data.csv"
#sources_file="input/3D_sources.csv"
#aux_par_file="input/3D_frac.csv"

data_file="input/generic2D_data.csv"
sources_file="input/fractionation2D_sources_delta.csv"
#aux_par_file=""
aux_par_file="input/fractionation2D_frac.csv"

model.load_from_fielnames(
        data_file=data_file,
        sources_file=sources_file,
        aux_file=aux_par_file,
        )


# set the output directory
model.set_outdir(out_dir="test_output")


#model.set_iterations(1e6, 1e3, 100)
model.set_iterations(1e5, 3e3, 700)

# Trun online plotting on/off
model.plotting_switch=True

# run model
model.run_model()
