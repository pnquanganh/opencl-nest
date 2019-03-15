import sys, getopt
import pprint
import re

model_directory = '../'
template_header = 'non_gsl_template.h'
template_source = 'non_gsl_template.cpp'
template_pre_post_gsl = 'non_gsl_emplate_pre_post_gsl.cpp'
template_build_graph = 'template_build_graph.cpp'
template_opencl = 'non_gsl_template_kernel.cl'

deliver_events_code = "deliver_events(h_buffer, h_buffer_mark, h_buffer_count, h_buffer_index, d_buffer_in_, d_buffer_out_, d_buffer_count, d_buffer_index);"

class Neuron_Variables:
    def __init__(self, var_type, host_id, gpu_id, is_scalar, array_size, is_output):
        self.var_type = var_type
        self.host_id = host_id
        self.gpu_id = gpu_id
        self.scalar = is_scalar
        self.array_size = array_size
        self.is_output = is_output

        self.host_buffer = 'h_' + self.gpu_id

    def device_model_parameters(self):
        return 'cl::Buffer ' + self.gpu_id + ';'

    def host_model_parameters(self):
        return self.var_type + '* ' + self.host_buffer + ';'

    def host_init(self):
        return ' , ' + self.host_buffer + '( NULL )'

    def host_free(self):        
        return 'if (' + self.host_buffer + ') delete[] ' + self.host_buffer + ';'

    def host_alloc(self):
        return '      ' + self.host_buffer + ' = new ' + self.var_type + ('[len];' if self.scalar else ('[len *' + str(self.array_size) + '];'))

    def device_alloc(self):
        return '      create(&gpu_context, &' + self.gpu_id + (', len' if self.scalar else ', len*' + str(self.array_size)) + '*sizeof(' + self.var_type + '));'

    def device_upload(self):
        return ('UPLOAD_1D_DATA' if self.scalar else 'UPLOAD_2D_DATA') + '(' + self.host_buffer + ', ' + self.host_id + ');'

    def device_finish_upload(self):
        return ('FINISH_1D_UPLOAD' if self.scalar else 'FINISH_2D_UPLOAD') + '(' + self.gpu_id + ', ' + self.host_buffer + ', ' + ('' if self.scalar else (str(self.array_size) + ', ')) + self.var_type + ');'

    def set_kernel(self):
        return '  set_kernel(' + self.gpu_id + ');'

    def device_download(self):
        return ('START_1D_DOWNLOAD' if self.scalar else 'START_2D_DOWNLOAD') + '(' + self.gpu_id + ',' + self.host_buffer + ', ' + ('' if self.scalar else (str(self.array_size) + ', ')) + self.var_type + ');'

    def device_finish_download(self):
        return ('DOWNLOAD_1D_DATA' if self.scalar else 'DOWNLOAD_2D_DATA') + '(' + self.host_id + ', ' + self.host_buffer + ');'

    def param_input_decl(self):
        return self.var_type + ' *' + self.gpu_id + ','
        # if self.is_output:
        #     return self.var_type + ' *' + self.gpu_buffer + ','
        # else:
        #     return self.var_type + ' ' + self.gpu_regs + ','

    def update_input_decl(self):
        return '__global ' + self.var_type + ' *' + self.gpu_id + ','

    def param_input_pass(self):
        return self.gpu_id + ','
        # if self.is_output:
        #     return self.gpu_buffer + ','
        # else:
        #     return self.gpu_regs + ','

    def param_to_regs(self):
        return self.var_type + ' ' + self.gpu_regs + ' = ' + self.gpu_buffer + '[tid];'

def process_parameter(str):
    pos = str.find('//')
    if pos != -1:
        str = str[:pos]
    pos = str.find('/*')
    if pos != -1:
        str = str[:pos]
    pos = str.find(';')
    if pos != -1:
        str = str[:pos]

    ret = str.strip().split()
    return ret

def process_states(str):
    pos = str.find(',')
    if pos != -1:
        str = str[:pos]

    pos = str.find('=')
    if pos != -1:
        str = str[:pos]

    return str.strip()

def gen_parameter_variable(param):
    return [param[0], param[1] + param[2], 'h_' + param[1] + param[2], param[1] + '.' + param[2], 'r_' + param[1] + param[2]]
    #return [param[0], 'P_' + param[1], 'h_P_' + param[1], 'P_.' + param[1], 'r_P_' + param[1]]

# def gen_host_variable(param):
#     return [param[0], 'h_P_' + param[1]]

def remove_comment(str):
    # i = str.find('/*')
    # while i != -1:
    #     j = str.find('*/', i)
    #     str = str[: i] + str[j + 2:]
    #     i = str.find('/*')
        
    regex = re.compile('^(\s*)//.*', re.MULTILINE)
    str = regex.sub('', str)
    # print str
    return str


def get_states(header_content):
    s = header_content.find('enum StateVecElems')
    if s == -1:
        return []
    s = header_content.find('{', s)
    e = header_content.find('}', s)

    list_states_str = header_content[s + 1:e - 1].splitlines()
    list_states = filter(None, [process_states(str) for str in list_states_str])
        
    return list_states

def gen_deliver_code(ring_buffer):
    code = deliver_events_code.replace('h_buffer', 'h_' + ring_buffer)
    code = code.replace('h_buffer_mark', 'h_' + ring_buffer + '_mark')
    code = code.replace('h_buffer_count', 'h_' + ring_buffer + '_count')
    code = code.replace('h_buffer_index', 'h_' + ring_buffer + '_index')
    code = code.replace('d_buffer_in_', 'd_' + ring_buffer + '_buf')
    code = code.replace('d_buffer_out_', 'd_' + ring_buffer)
    code = code.replace('d_buffer_count', 'd_' + ring_buffer + '_count')
    code = code.replace('d_buffer_index', 'd_' + ring_buffer + '_index')
    return code
    
def gen_gpu_header(model_name, parameter_variables, list_ring_buffers, parsed_content):
    
    with open(template_header, 'r') as file:
        gen_header = file.read()

    gen_header = gen_header.replace('model_name', model_name)
    gen_header = gen_header.replace('MODEL_NAME', model_name.upper())
    dev_buffers = "\n".join([var.device_model_parameters() for k, var in parameter_variables.iteritems()])
    gen_header = gen_header.replace('/* DEVICE BUFFERS */', dev_buffers)
    host_buffers = "\n".join([var.host_model_parameters() for k, var in parameter_variables.iteritems()])
    gen_header = gen_header.replace('/* HOST BUFFERS */', host_buffers)

    decl_host_ring_buffers = "\n".join(['double* h_' + rb + ';' for rb in list_ring_buffers])
    # decl_host_ring_buffers += "\n" + "\n".join(['int* h_' + rb + '_mark;' for rb in list_ring_buffers])
    # decl_host_ring_buffers += "\n" + "\n".join(['int* h_' + rb + '_count;' for rb in list_ring_buffers])
    # decl_host_ring_buffers += "\n" + "\n".join(['int* h_' + rb + '_index;' for rb in list_ring_buffers])
    gen_header = gen_header.replace('/*HOST RING BUFFER*/', decl_host_ring_buffers)

    decl_device_ring_buffers = "\n".join(['cl::Buffer d_' + rb + '_buf;' for rb in list_ring_buffers])
    decl_device_ring_buffers += '\n' + "\n".join(['cl::Buffer d_' + rb + ';' for rb in list_ring_buffers])
    # decl_device_ring_buffers += '\n' + "\n".join(['cl::Buffer d_' + rb + '_count;' for rb in list_ring_buffers])
    # decl_device_ring_buffers += '\n' + "\n".join(['cl::Buffer d_' + rb + '_index;' for rb in list_ring_buffers])
    gen_header = gen_header.replace('/*DEVICE RING BUFFER*/', decl_device_ring_buffers)

    # dev_param_dec = "\n".join([var.device_model_parameters() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_header = gen_header.replace('/* DEVICE OUTPUT VAR */', dev_param_dec)
    # host_param_dec = "\n".join([var.host_model_parameters() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_header = gen_header.replace('/* HOST OUTPUT VAR */', host_param_dec)

    handle_func = []
    i = parsed_content.find('/*HANDLE START*/')
    while i != -1:
        e = parsed_content.find('{', i)
        handle_func = handle_func + [parsed_content[i + 18: e - 1]]
        i = parsed_content.find('/*HANDLE START*/', e)

    handle_func_decl = "\n".join(['void ' + f + ';' for f in handle_func])
    gen_header = gen_header.replace('/*HANDLE FUNCTIONS*/', handle_func_decl)
    
    gpu_header_file = '../' + model_name + '_gpu.h'
    with open(gpu_header_file, 'w') as file:
        file.write(gen_header)

def gen_gpu_source(model_name, parameter_variables, list_ring_buffers, parsed_content):

    with open(template_source, 'r') as file:
        gen_source = file.read()

    gen_source = gen_source.replace('model_name', model_name)
    #gen_header.replace('MODEL_NAME', model_name.upper())

    host_buffers_init = "\n".join([var.host_init() for k, var in parameter_variables.iteritems()])
    gen_source = gen_source.replace('/* HOST BUFFERS INIT */', host_buffers_init)

    host_ring_buffers_init = "\n".join([', h_' + rb + '( NULL )' for rb in list_ring_buffers])
    # host_ring_buffers_init += "\n" + "\n".join([', h_' + rb + '_mark( NULL )' for rb in list_ring_buffers])
    # host_ring_buffers_init += "\n" + "\n".join([', h_' + rb + '_count( NULL )' for rb in list_ring_buffers])
    # host_ring_buffers_init += "\n" + "\n".join([', h_' + rb + '_index( NULL )' for rb in list_ring_buffers])
    gen_source = gen_source.replace('/* HOST RING BUFFERS INIT */', host_ring_buffers_init)
    
    # host_param_init = "\n".join([var.host_init() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* HOST OUTPUT VAR INIT */', host_param_init)

    host_buffers_free = "\n".join([var.host_free() for k, var in parameter_variables.iteritems()])
    gen_source = gen_source.replace('/* HOST BUFFERS FREE */', host_buffers_free)

    host_ring_buffers_free = "\n".join(['if (h_' + rb + ') delete[] h_' + rb + ';' for rb in list_ring_buffers])
    # host_ring_buffers_free += "\n" + "\n".join(['if (h_' + rb + '_mark) delete[] h_' + rb + '_mark;' for rb in list_ring_buffers])
    # host_ring_buffers_free += "\n" + "\n".join(['if (h_' + rb + '_count) delete[] h_' + rb + '_count;' for rb in list_ring_buffers])
    # host_ring_buffers_free += "\n" + "\n".join(['if (h_' + rb + '_index) delete[] h_' + rb + '_index;' for rb in list_ring_buffers])
    gen_source = gen_source.replace('/* HOST RING BUFFERS FREE */', host_ring_buffers_free)

    # host_param_free = "\n".join([var.host_free() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* HOST OUTPUT VAR FREE */', host_param_free)

    host_buffers_alloc = "\n".join([var.host_alloc() for k, var in parameter_variables.iteritems()])
    gen_source = gen_source.replace('/* HOST BUFFERS ALLOC */', host_buffers_alloc)

    host_ring_buffers_alloc = "\n".join(['h_' + rb + ' = new double[ring_buffer_size];' for rb in list_ring_buffers])
    # host_ring_buffers_alloc += "\n" + "\n".join(['h_' + rb + '_mark = new int[ring_buffer_size];' for rb in list_ring_buffers])
    # host_ring_buffers_alloc += "\n" + "\n".join(['h_' + rb + '_count = new int[total_num_nodes];' for rb in list_ring_buffers])
    # host_ring_buffers_alloc += "\n" + "\n".join(['h_' + rb + '_index = new int[ring_buffer_size];' for rb in list_ring_buffers])
    gen_source = gen_source.replace('/* HOST RING BUFFERS ALLOC */', host_ring_buffers_alloc)
    
    # host_param_alloc = "\n".join([var.host_alloc() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* HOST OUTPUT VAR ALLOC */', host_param_alloc)

    device_buffers_alloc = "\n".join([var.device_alloc() for k, var in parameter_variables.iteritems()])
    gen_source = gen_source.replace('/* DEVICE BUFFERS ALLOC */', device_buffers_alloc)

    device_ring_buffers_alloc = "\n".join(['create(&gpu_context, &d_' + rb + ', ring_buffer_size*sizeof(double));' for rb in list_ring_buffers])
    # device_ring_buffers_alloc = device_ring_buffers_alloc + "\n".join(['create(&gpu_context, &d_' + rb + '_buf, ring_buffer_size*sizeof(double));' for rb in list_ring_buffers])
    # device_ring_buffers_alloc = device_ring_buffers_alloc + '\n' + "\n".join(['create(&gpu_context, &d_' + rb + '_count, total_num_nodes*sizeof(int));' for rb in list_ring_buffers])
    # device_ring_buffers_alloc = device_ring_buffers_alloc + '\n' + "\n".join(['create(&gpu_context, &d_' + rb + '_index, ring_buffer_size*sizeof(int));' for rb in list_ring_buffers])

    # device_ring_buffers_alloc = device_ring_buffers_alloc + '\n' + "\n".join(['fill_buffer_zero_double(&gpu_context, d_' + rb + '_buf, ring_buffer_size * sizeof(double));' for rb in list_ring_buffers])
    device_ring_buffers_alloc = device_ring_buffers_alloc + '\n' + "\n".join(['fill_buffer_zero_double(&gpu_context, d_' + rb + ', ring_buffer_size * sizeof(double));' for rb in list_ring_buffers])
    
    gen_source = gen_source.replace('/* DEVICE RING BUFFERS ALLOC */', device_ring_buffers_alloc)

    # device_param_alloc = "\n".join([var.device_alloc() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* DEVICE OUTPUT VAR ALLOC */', device_param_alloc)

    device_buffers_upload = "\n".join([var.device_upload() for k, var in parameter_variables.iteritems()])
    gen_source = gen_source.replace('/* DEVICE BUFFERS UPLOAD */', device_buffers_upload)
    # device_param_upload = "\n".join([var.device_upload() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* DEVICE OUTPUT VAR UPLOAD */', device_param_upload)

    device_buffers_fin_upload = "\n".join([var.device_finish_upload() for k, var in parameter_variables.iteritems()])
    gen_source = gen_source.replace('/* DEVICE BUFFERS FINISH UPLOAD */', device_buffers_fin_upload)
    # device_param_fin_upload = "\n".join([var.device_finish_upload() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* DEVICE OUTPUT VAR FINISH UPLOAD */', device_param_fin_upload)

    param_set_arg = "\n".join([var.set_kernel() for k,var in parameter_variables.iteritems() if not var.is_output])
    gen_source = gen_source.replace('/* MODEL BUFFERS SET ARG */', param_set_arg)

    ring_buffer_set_arg = "\n".join(['  set_kernel(d_' + rb + ');' for rb in list_ring_buffers])
    # ring_buffer_set_arg += "\n" + "\n".join(['  set_kernel(d_' + rb + '_count);' for rb in list_ring_buffers])
    # ring_buffer_set_arg += "\n" + "\n".join(['  set_kernel(d_' + rb + '_index);' for rb in list_ring_buffers])
    gen_source = gen_source.replace('/* RING BUFFERS SET ARG */', ring_buffer_set_arg)
    
    param_set_arg = "\n".join([var.set_kernel() for k, var in parameter_variables.iteritems() if var.is_output])
    gen_source = gen_source.replace('/* OUTPUT VAR SET ARG */', param_set_arg)

    host_ring_buffers_clear = "\n".join(['h_' + rb + '[i] = 0.0;' for rb in list_ring_buffers])
    # host_ring_buffers_clear += "\n" + "\n".join(['h_' + rb + '_mark[i] = 0;' for rb in list_ring_buffers])

    gen_source = gen_source.replace('/* HOST RING BUFFERS CLEAR */', host_ring_buffers_clear)

    # device_download = "\n".join([var.device_download() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* DEVICE OUTPUT VAR DOWNLOAD */', device_download)

    # device_fin_download = "\n".join([var.device_finish_download() for k, var in parameter_variables.iteritems() if var.is_output])
    # gen_source = gen_source.replace('/* DEVICE OUTPUT VAR FINISH DOWNLOAD */', device_fin_download)

    handle_func = []
    i = parsed_content.find('/*HANDLE START*/')
    while i != -1:
        e = parsed_content.find('/*HANDLE END*/', i)
        handle_body = parsed_content[i + 16: e]
        o = handle_body.find('{')
        handle_body = handle_body[:o] + '{}'
        handle_func = handle_func + [handle_body]
        i = parsed_content.find('/*HANDLE START*/', e)

    handle_func_decl = "\n".join(['void nest::' + model_name + '_gpu' + f + ';' for f in handle_func])
    gen_source = gen_source.replace('/*HANDLE DEFINITION*/', handle_func_decl)

    deliver_code = "\n".join([gen_deliver_code(rb) for rb in list_ring_buffers])
    gen_source = gen_source.replace('/*DELIVER EVENT CODE*/', deliver_code)

    gpu_source_file = '../' + model_name + '_gpu.cpp'
    with open(gpu_source_file, 'w') as file:
        file.write(gen_source)

    # source_file = model_directory + model_name + '.cpp'
    # with open(source_file, 'r') as file:
    #     source_content = file.read()

def gen_build_graph(model_name):
    with open(template_build_graph, 'r') as file:
        build_graph_source = file.read()

    build_graph_source = build_graph_source.replace('model_name', model_name)
        
    filename = '../' + model_name + '_build_graph.cpp'
    with open(filename, 'w') as file:
        file.write(build_graph_source)
    
def process_array_index(dynamic_function):
    s = dynamic_function.find('[')
    while s != -1:
        e = dynamic_function.find(']', s)
        array_indx = dynamic_function[s + 1: e - 1].strip()
        dynamic_function = dynamic_function[:s + 1] + 'num_nodes * (' + array_indx + ') + tid' + dynamic_function[e:]
        s = dynamic_function.find('[', s + 1)

    return dynamic_function

def process_gpu_code(source_code, parameter_variables):
    source_code = source_code.replace('&', '')
    source_code = source_code.replace('node.', '')
    source_code = source_code.replace('S::', '')
    source_code = source_code.replace('State_::', '')
    source_code = source_code.replace('std::', '')
    
    for k, s in parameter_variables.iteritems():
        if not s.is_output:
            source_code = source_code.replace(s.cpu_regs, s.gpu_regs)
            
    source_code = process_array_index(source_code)
            
    for k, s in parameter_variables.iteritems():
        if s.is_output:
            source_code = source_code.replace(s.cpu_regs, s.gpu_buffer + ('[tid]' if s.scalar else ''))
            
    return source_code

    
def gen_opencl(model_name, states, parameter_variables, parsed_content, list_ring_buffers):
    with open(template_opencl, 'r') as file:
        gen_opencl_content = file.read()

    define_states = "\n".join(['#define ' + s + ' ' + str(idx) for idx, s in enumerate(states)])
    gen_opencl_content = gen_opencl_content.replace('/* DEFINE STATES */', define_states)

    parameter_input_decl = "\n".join([s.param_input_decl() for k, s in parameter_variables.iteritems() if not s.is_output])
    gen_opencl_content = gen_opencl_content.replace('/* MODEL PARAMETER INPUT DECL */', parameter_input_decl)

    parameter_update_input_decl = "\n".join([s.update_input_decl() for k, s in parameter_variables.iteritems() if not s.is_output])
    gen_opencl_content = gen_opencl_content.replace('/* MODEL PARAMETER UPDATE INPUT DECL */', parameter_update_input_decl)

    parameter_input_pass = "\n".join([s.param_input_pass() for k, s in parameter_variables.iteritems() if not s.is_output])
    gen_opencl_content = gen_opencl_content.replace('/* MODEL PARAMETER INPUT PASS */', parameter_input_pass)
    # global gsl_loop
    # gsl_loop = gsl_loop.replace('/* MODEL PARAMETER INPUT PASS */', parameter_input_pass)

    # get_params_to_regs = "\n".join([s.param_to_regs() for k, s in parameter_variables.iteritems() if not s.is_output and s.scalar])
    # gen_opencl_content = gen_opencl_content.replace('/* GET MODEL PARAMETER TO REGS */', get_params_to_regs)

    s = parsed_content.find('/*DYNAMICS FUNCTION START*/')
    e = parsed_content.find('/*DYNAMICS FUNCTION END*/')

    dynamic_function = parsed_content[s + 27:e];

    # dynamic_function = remove_comment(dynamic_function)

    # dynamic_function = process_gpu_code(dynamic_function, parameter_variables)
    
    gen_opencl_content = gen_opencl_content.replace('/* DEFINE DYNAMICS FUNCTION */', dynamic_function)

    ring_buffer_decl = "\n".join(['__global double *' + rb + ',' for rb in list_ring_buffers])
    gen_opencl_content = gen_opencl_content.replace('/* RING BUFFER INPUT DECL */', ring_buffer_decl)
    
    s = parsed_content.find('/*GPU START*/')
    if s != -1:
        e = parsed_content.find('/*GPU END*/', s)
        update_body = parsed_content[s + 13: e]
        update_body = update_body.replace('/*SPIKE SEND*/', 'spike_count[tid]++;\n')
        gen_opencl_content = gen_opencl_content.replace('/* UPDATE BODY */', update_body)
        
    filename = '../kernel/' + model_name + '.cl'
    with open(filename, 'w') as file:
        file.write(gen_opencl_content)
    
def main(argv):
    pp = pprint.PrettyPrinter(indent=4)
    try:
        opts, args = getopt.getopt(argv, '')

        if not args:
            print 'Missing model name'
            sys.exit(2)

        input_lines = [line for line in sys.stdin]
        list_param = dict()
        num_ele = int(input_lines[0])
        
        l = 1
        for i in range(num_ele):
            var_type = input_lines[l].strip()
            host_id = input_lines[l + 1].strip()
            gpu_id = input_lines[l + 2].strip()
            is_scalar = True if input_lines[l + 3].strip() == "True" else False
            array_size = int(input_lines[l + 4].strip())
            is_output = True if input_lines[l + 5].strip() == "True" else False
        
            l = l + 6
            param_ele = Neuron_Variables(var_type, host_id, gpu_id, is_scalar, array_size, is_output)
            list_param[host_id] = param_ele

        
        num_rb = int(input_lines[l].strip())
        l = l + 1
        list_ring_buffers = [rb.strip() for rb in input_lines[l:l+num_rb]]
        l = l + num_rb

        parsed_content = ''.join(input_lines[l:])

        model_name = args[0]
        header_file = model_directory + model_name + '.h'
        with open(header_file, 'r') as file:
            header_content = file.read()

        # source_file = model_directory + model_name + '.cpp'
        # with open(source_file, 'r') as file:
        #     source_content = file.read()

        states = get_states(header_content)

        # #parameter_variables = [gen_parameter_variable(param) for param in model_parameters]
        # #host_variables = [gen_host_variable(param) for param in model_parameters]

        # #pp.pprint(parameter_variables)
        # #pp.pprint(host_variables)

        # parameter_variables = gen_pre_post_gsl(model_name, source_content, parameter_variables)
        gen_build_graph(model_name)
        
        gen_gpu_header(model_name, list_param, list_ring_buffers, parsed_content)
        gen_gpu_source(model_name, list_param, list_ring_buffers, parsed_content)

        gen_opencl(model_name, states, list_param, parsed_content, list_ring_buffers)
        
    except getopt.GetoptError:
        print 'Missing model name'
        sys.exit(2)
    return

if __name__ == '__main__':
    main(sys.argv[1:])
