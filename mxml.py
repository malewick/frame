import sys
import NDimModel

model = NDimModel.Model()

model.load_from_xml("/home/maciej/cern/izotopy/input/3D.xml")

model.set_outdir(out_dir="test_output")

#model.set_iterations(1e6, 1e3, 100)
model.set_iterations(1e5, 1e3, 300)

model.plotting_switch=False

model.run_model()
