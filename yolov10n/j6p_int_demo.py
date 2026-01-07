from hmct.api import load_model
from hbdk4.compiler.onnx import export
from hbdk4.compiler import convert,compile,visualize

def remove_op_by_ioname(func, io_name=None):
    for loc in  func.outputs + func.inputs:
        if not loc.is_removable[0]:
            if io_name == loc.name:
                raise ValueError(f"Failed when deleting {io_name} ,which id unremovable")
            continue
        attached_op = loc.get_attached_op[0]
        removed = None
        output_name = attached_op.outputs[0].name
        input_name = attached_op.inputs[0].name

        if io_name in [output_name, input_name]:
            removed, diagnostic = loc.remove_attached_op()
        if removed is True:
            print(f"Remove node {io_name} successfully",flush=True)
        if removed is False:
            raise ValueError(
                f"Failed when deleting {attached_op.name} operator,"
                f"error: {diagnostic}")

onnx_model = load_model("yolov10n_output_s600/yolov10n_640x640_nv12_ptq_model.onnx")
hb_ptq_model = export(onnx_model)
hb_ptq_model._integer_conv = True
hb_quantized_model = convert(hb_ptq_model,"nash-p",advice=True)
func = hb_quantized_model.functions[0]
remove_op_by_ioname(func=func, io_name="471")
remove_op_by_ioname(func=func, io_name="493")
remove_op_by_ioname(func=func, io_name="515")

compile(hb_quantized_model,"./yolov10n_640x640_nv12.hbm", "nash-p")
