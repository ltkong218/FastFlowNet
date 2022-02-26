//
// Created by jerry on 2022/2/26.
//
#include "NvInfer.h"

#include "correlationPlugin.h"

#include <cassert>
#include <cstring>
#include <vector>
#include <stdio.h>

using namespace nvinfer1;
using nvinfer1::plugin::CorrelationPlugin;
using nvinfer1::plugin::CorrelationPluginCreator;

// plugin specific constants
namespace
{
static const char* CORRELATION_PLUGIN_VERSION{"1"};
static const char* CORRELATION_PLUGIN_NAME{"Correlation"}; // creator will concat plugintype and namespace
static const char* CORRELATION_PLUGIN_NAMESPACE{""};
} // namespace

// Static class fields initialization
PluginFieldCollection CorrelationPluginCreator::mFC{};
std::vector<PluginField> CorrelationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CorrelationPluginCreator);

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

CorrelationPlugin::CorrelationPlugin(const std::string name, const void* serial_buf, size_t serial_size)
    : mLayerName(name)
{
    const char* d = reinterpret_cast<const char*>(serial_buf);
    const char* a = d;
    mQueryChannel = readFromBuffer<size_t>(d);
    mQueryHeight = readFromBuffer<size_t>(d);
    mQueryWidth = readFromBuffer<size_t>(d);
    mOutputChannel = readFromBuffer<size_t>(d);
    mPatchHeight = readFromBuffer<size_t>(d);
    mPatchWidth = readFromBuffer<size_t>(d);
    mDilation = readFromBuffer<size_t>(d);
    mType = readFromBuffer<DataType>(d);
    ASSERT(d == a + sizeof(size_t) * 7 + sizeof(DataType));
}

CorrelationPlugin::CorrelationPlugin(const std::string name): mLayerName(name){
}

// for clone
CorrelationPlugin::CorrelationPlugin(const std::string name, int queryChannel, int queryHeight,
    int queryWidth, int outputChannel, int patchHeight, int patchWidth,int dilation, DataType type)
    : mLayerName(name)
    , mQueryChannel(queryChannel)
    , mQueryHeight(queryHeight)
    , mQueryWidth(queryWidth)
    , mOutputChannel(outputChannel)
    , mPatchHeight(patchHeight)
    , mPatchWidth(patchWidth)
    , mDilation(dilation)
    , mType(type)
{
}

CorrelationPlugin::~CorrelationPlugin() {}

const char* CorrelationPlugin::getPluginType() const noexcept
{
    return CORRELATION_PLUGIN_NAME;
}

const char* CorrelationPlugin::getPluginVersion() const noexcept
{
    return CORRELATION_PLUGIN_VERSION;
}

int CorrelationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs CorrelationPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Validate input arguments
    assert(inputs[0].nbDims == 4);
    assert(inputs[1].nbDims == 4);

    // return N, C, H_g, W_g
    DimsExprs output(inputs[0]);
    output.d[1] = exprBuilder.constant(mOutputChannel);
    output.d[2] = inputs[1].d[2];
    output.d[3] = inputs[1].d[3];
    return output;
}

int CorrelationPlugin::initialize() noexcept
{
    return 0;
}

size_t CorrelationPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int CorrelationPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int status = -1;

    //    GridSamplerDataType dataType = (mType == DataType::kFLOAT ? GridSamplerDataType::GFLOAT : GridSamplerDataType::GHALF);
    CorrelationDataType dataType = (inputDesc->type == DataType::kFLOAT ? CorrelationDataType::GFLOAT : CorrelationDataType::GHALF);

    status = correlation_cuda(mBatch,inputs[0],inputs[1],outputs[0],
                                            mQueryChannel,mQueryHeight,mQueryWidth,mQueryHeight,mQueryWidth,
                                            mQueryChannel*mQueryHeight*mQueryWidth,mQueryHeight*mQueryWidth,mQueryWidth,1,
                                            mPatchHeight*mPatchWidth*mQueryHeight*mQueryWidth,mQueryHeight*mQueryWidth,mQueryWidth,1,
                                            1,1,mPatchHeight,mPatchWidth,int((mPatchHeight-1)/2),int((mPatchWidth-1)/2),mDilation,
                                            dataType,stream);

    return status;
}

size_t CorrelationPlugin::getSerializationSize() const noexcept
{
    return sizeof(size_t) * 7 + sizeof(DataType);
}

void CorrelationPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    writeToBuffer<size_t>(d, mQueryChannel);
    writeToBuffer<size_t>(d, mQueryHeight);
    writeToBuffer<size_t>(d, mQueryWidth);
    writeToBuffer<size_t>(d, mOutputChannel);
    writeToBuffer<size_t>(d, mPatchHeight);
    writeToBuffer<size_t>(d, mPatchWidth);
    writeToBuffer<size_t>(d, mDilation);
    writeToBuffer<DataType>(d, mType);
    ASSERT(d == a + getSerializationSize());
}

bool CorrelationPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

    bool condition = inOut[pos].format == TensorFormat::kLINEAR;

    condition &= inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF;
    condition &= inOut[pos].type == inOut[0].type;
    return condition;
}

void CorrelationPlugin::terminate() noexcept {}

void CorrelationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* CorrelationPlugin::clone() const noexcept
{
    auto plugin
        = new CorrelationPlugin(mLayerName, mQueryChannel,mQueryHeight,mQueryWidth,
                                mOutputChannel,mPatchHeight,mPatchWidth,mDilation,mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void CorrelationPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* CorrelationPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index.
DataType CorrelationPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // one outputs
    ASSERT(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}


void CorrelationPlugin::configurePlugin(
    const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    // we only support 2d grid sampler now.
    ASSERT(inputs[0].desc.dims.nbDims == 4);
    ASSERT(inputs[1].desc.dims.nbDims == 4);

    mBatch = inputs[0].desc.dims.d[0];
    mQueryChannel = inputs[0].desc.dims.d[1];
    mQueryHeight = inputs[0].desc.dims.d[2];
    mQueryWidth = inputs[0].desc.dims.d[3];

    ASSERT(mBatch == inputs[1].desc.dims.d[0]);
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void CorrelationPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void CorrelationPlugin::detachFromContext() noexcept {}

CorrelationPluginCreator::CorrelationPluginCreator()
{
    setPluginNamespace(CORRELATION_PLUGIN_NAMESPACE);
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

CorrelationPluginCreator::~CorrelationPluginCreator() {}

const char* CorrelationPluginCreator::getPluginName() const noexcept
{
    return CORRELATION_PLUGIN_NAME;
}

const char* CorrelationPluginCreator::getPluginVersion() const noexcept
{
    return CORRELATION_PLUGIN_VERSION;
}

const PluginFieldCollection* CorrelationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* CorrelationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int interpolationMode = 0, paddingMode = 0, alignCorners = 0;

    for (int i = 0; i < nbFields; ++i)
    {
        ASSERT(fields[i].type == PluginFieldType::kINT32);

        if (!strcmp(fields[i].name, "interpolation_mode"))
        {
            interpolationMode = *(reinterpret_cast<const int*>(fields[i].data));
        }

        if (!strcmp(fields[i].name, "padding_mode"))
        {
            paddingMode = *(reinterpret_cast<const int*>(fields[i].data));
        }

        if (!strcmp(fields[i].name, "align_corners"))
        {
            alignCorners = *(reinterpret_cast<const int*>(fields[i].data));
        }
    }

    auto plugin = new CorrelationPlugin(name);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* CorrelationPluginCreator::deserializePlugin (
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    auto plugin = new CorrelationPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
