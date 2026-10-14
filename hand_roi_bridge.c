#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>

#define HAND_OUTPUTS 4
#define HAND_LANDMARK_VALUES 63
#define GESTURE_SEQUENCE_LENGTH 8
#define GESTURE_LANDMARKS 21
#define GESTURE_CHANNELS 3
#define GESTURE_OUTPUTS 23

typedef struct {
  hbPackedDNNHandle_t packed;
  hbDNNHandle_t model;
  hbSysMem y;
  hbSysMem uv;
  int allocated_width;
  int allocated_height;
  int stride;
  char error[256];
} HandRoiContext;

typedef struct {
  hbPackedDNNHandle_t packed;
  hbDNNHandle_t model;
  hbDNNTensor input;
  hbDNNTensor output;
  int ready;
  char error[256];
} GestureContext;

static int fail(HandRoiContext *ctx, int code, const char *where) {
  if (ctx != NULL) {
    snprintf(ctx->error, sizeof(ctx->error), "%s failed (%d)", where, code);
  }
  return code == 0 ? -1 : code;
}

static int aligned16(int value) { return (value + 15) & ~15; }

static void free_input(HandRoiContext *ctx) {
  if (ctx->y.virAddr != NULL) {
    hbSysFreeMem(&ctx->y);
  }
  if (ctx->uv.virAddr != NULL) {
    hbSysFreeMem(&ctx->uv);
  }
  memset(&ctx->y, 0, sizeof(ctx->y));
  memset(&ctx->uv, 0, sizeof(ctx->uv));
  ctx->allocated_width = 0;
  ctx->allocated_height = 0;
  ctx->stride = 0;
}

static int ensure_input(HandRoiContext *ctx, int width, int height) {
  if (ctx->allocated_width == width && ctx->allocated_height == height) {
    return 0;
  }
  free_input(ctx);
  ctx->stride = aligned16(width);
  int ret = hbSysAllocCachedMem(&ctx->y, (uint32_t)(ctx->stride * height));
  if (ret != 0) {
    return fail(ctx, ret, "hbSysAllocCachedMem(Y)");
  }
  ret = hbSysAllocCachedMem(&ctx->uv,
                            (uint32_t)(ctx->stride * height / 2));
  if (ret != 0) {
    hbSysFreeMem(&ctx->y);
    memset(&ctx->y, 0, sizeof(ctx->y));
    return fail(ctx, ret, "hbSysAllocCachedMem(UV)");
  }
  ctx->allocated_width = width;
  ctx->allocated_height = height;
  return 0;
}

void *hand_roi_create(const char *model_path) {
  HandRoiContext *ctx = (HandRoiContext *)calloc(1, sizeof(*ctx));
  if (ctx == NULL) {
    return NULL;
  }
  const char *files[1] = {model_path};
  int ret = hbDNNInitializeFromFiles(&ctx->packed, files, 1);
  if (ret != 0) {
    fail(ctx, ret, "hbDNNInitializeFromFiles");
    return ctx;
  }
  const char **names = NULL;
  int32_t count = 0;
  ret = hbDNNGetModelNameList(&names, &count, ctx->packed);
  if (ret != 0 || count < 1) {
    fail(ctx, ret == 0 ? -1 : ret, "hbDNNGetModelNameList");
    return ctx;
  }
  ret = hbDNNGetModelHandle(&ctx->model, ctx->packed, names[0]);
  if (ret != 0) {
    fail(ctx, ret, "hbDNNGetModelHandle");
    return ctx;
  }
  snprintf(ctx->error, sizeof(ctx->error), "ok");
  return ctx;
}

const char *hand_roi_last_error(void *opaque) {
  HandRoiContext *ctx = (HandRoiContext *)opaque;
  return ctx == NULL ? "null hand ROI context" : ctx->error;
}

int hand_roi_ready(void *opaque) {
  HandRoiContext *ctx = (HandRoiContext *)opaque;
  return ctx != NULL && ctx->model != NULL;
}

int hand_roi_infer(void *opaque, const uint8_t *nv12, int width, int height,
                   float *landmarks63, float *score, float *handedness) {
  HandRoiContext *ctx = (HandRoiContext *)opaque;
  if (ctx == NULL || ctx->model == NULL || nv12 == NULL ||
      landmarks63 == NULL || score == NULL || handedness == NULL) {
    return fail(ctx, -1, "invalid argument");
  }
  if (width < 18 || height < 18 || width > 784 || height > 784 ||
      (width & 1) || (height & 1)) {
    return fail(ctx, -1, "source size must be even and within 18..784");
  }
  int ret = ensure_input(ctx, width, height);
  if (ret != 0) {
    return ret;
  }

  const uint8_t *src_y = nv12;
  const uint8_t *src_uv = nv12 + width * height;
  uint8_t *dst_y = (uint8_t *)ctx->y.virAddr;
  uint8_t *dst_uv = (uint8_t *)ctx->uv.virAddr;
  for (int row = 0; row < height; ++row) {
    memcpy(dst_y + row * ctx->stride, src_y + row * width, (size_t)width);
  }
  for (int row = 0; row < height / 2; ++row) {
    memcpy(dst_uv + row * ctx->stride, src_uv + row * width, (size_t)width);
  }
  ret = hbSysFlushMem(&ctx->y, HB_SYS_MEM_CACHE_CLEAN);
  if (ret != 0) {
    return fail(ctx, ret, "hbSysFlushMem(Y)");
  }
  ret = hbSysFlushMem(&ctx->uv, HB_SYS_MEM_CACHE_CLEAN);
  if (ret != 0) {
    return fail(ctx, ret, "hbSysFlushMem(UV)");
  }

  hbDNNTensor input;
  memset(&input, 0, sizeof(input));
  ret = hbDNNGetInputTensorProperties(&input.properties, ctx->model, 0);
  if (ret != 0) {
    return fail(ctx, ret, "hbDNNGetInputTensorProperties");
  }
  input.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;
  input.properties.validShape.dimensionSize[2] = height;
  input.properties.validShape.dimensionSize[3] = width;
  input.properties.alignedShape.dimensionSize[2] = height;
  input.properties.alignedShape.dimensionSize[3] = ctx->stride;
  input.sysMem[0] = ctx->y;
  input.sysMem[1] = ctx->uv;

  hbDNNTensor outputs[HAND_OUTPUTS];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < HAND_OUTPUTS; ++i) {
    ret = hbDNNGetOutputTensorProperties(&outputs[i].properties, ctx->model, i);
    if (ret != 0) {
      for (int j = 0; j < i; ++j) hbSysFreeMem(&outputs[j].sysMem[0]);
      return fail(ctx, ret, "hbDNNGetOutputTensorProperties");
    }
    ret = hbSysAllocCachedMem(&outputs[i].sysMem[0],
                              (uint32_t)outputs[i].properties.alignedByteSize);
    if (ret != 0) {
      for (int j = 0; j < i; ++j) hbSysFreeMem(&outputs[j].sysMem[0]);
      return fail(ctx, ret, "hbSysAllocCachedMem(output)");
    }
  }

  hbDNNRoi roi = {0, 0, width - 1, height - 1};
  hbDNNInferCtrlParam ctrl;
  memset(&ctrl, 0, sizeof(ctrl));
  ctrl.bpuCoreId = HB_BPU_CORE_ANY;
  ctrl.dspCoreId = HB_DSP_CORE_ANY;
  ctrl.priority = HB_DNN_PRIORITY_LOWEST;
  hbDNNTaskHandle_t task = NULL;
  hbDNNTensor *output_ptr = outputs;
  ret = hbDNNRoiInfer(&task, &output_ptr, &input, &roi, 1, ctx->model, &ctrl);
  if (ret == 0) {
    ret = hbDNNWaitTaskDone(task, 1000);
  }
  if (task != NULL) {
    hbDNNReleaseTask(task);
  }
  if (ret == 0) {
    for (int i = 0; i < HAND_OUTPUTS; ++i) {
      int flush_ret = hbSysFlushMem(&outputs[i].sysMem[0],
                                    HB_SYS_MEM_CACHE_INVALIDATE);
      if (flush_ret != 0) {
        ret = flush_ret;
        break;
      }
    }
  }
  if (ret == 0) {
    const float *point_data = (const float *)outputs[0].sysMem[0].virAddr;
    int stride = outputs[0].properties.stride[1] / (int)sizeof(float);
    if (stride < 1) stride = 1;
    for (int i = 0; i < HAND_LANDMARK_VALUES; ++i) {
      landmarks63[i] = point_data[i * stride];
    }
    *score = ((const float *)outputs[1].sysMem[0].virAddr)[0];
    *handedness = ((const float *)outputs[2].sysMem[0].virAddr)[0];
    snprintf(ctx->error, sizeof(ctx->error), "ok");
  } else {
    fail(ctx, ret, "hbDNNRoiInfer/hbDNNWaitTaskDone");
  }
  for (int i = 0; i < HAND_OUTPUTS; ++i) {
    hbSysFreeMem(&outputs[i].sysMem[0]);
  }
  return ret;
}

void hand_roi_destroy(void *opaque) {
  HandRoiContext *ctx = (HandRoiContext *)opaque;
  if (ctx == NULL) return;
  free_input(ctx);
  if (ctx->packed != NULL) {
    hbDNNRelease(ctx->packed);
  }
  free(ctx);
}

static int gesture_fail(GestureContext *ctx, int code, const char *where) {
  if (ctx != NULL) {
    snprintf(ctx->error, sizeof(ctx->error), "%s failed (%d)", where, code);
  }
  return code == 0 ? -1 : code;
}

static void gesture_release_tensors(GestureContext *ctx) {
  if (ctx->input.sysMem[0].virAddr != NULL) {
    hbSysFreeMem(&ctx->input.sysMem[0]);
  }
  if (ctx->output.sysMem[0].virAddr != NULL) {
    hbSysFreeMem(&ctx->output.sysMem[0]);
  }
  memset(&ctx->input, 0, sizeof(ctx->input));
  memset(&ctx->output, 0, sizeof(ctx->output));
}

void *gesture_model_create(const char *model_path) {
  GestureContext *ctx = (GestureContext *)calloc(1, sizeof(*ctx));
  if (ctx == NULL) {
    return NULL;
  }
  const char *files[1] = {model_path};
  int ret = hbDNNInitializeFromFiles(&ctx->packed, files, 1);
  if (ret != 0) {
    gesture_fail(ctx, ret, "hbDNNInitializeFromFiles");
    return ctx;
  }
  const char **names = NULL;
  int32_t count = 0;
  ret = hbDNNGetModelNameList(&names, &count, ctx->packed);
  if (ret != 0 || count < 1) {
    gesture_fail(ctx, ret == 0 ? -1 : ret, "hbDNNGetModelNameList");
    return ctx;
  }
  ret = hbDNNGetModelHandle(&ctx->model, ctx->packed, names[0]);
  if (ret != 0) {
    gesture_fail(ctx, ret, "hbDNNGetModelHandle");
    return ctx;
  }
  ret = hbDNNGetInputTensorProperties(&ctx->input.properties, ctx->model, 0);
  if (ret != 0) {
    gesture_fail(ctx, ret, "hbDNNGetInputTensorProperties");
    return ctx;
  }
  ret = hbDNNGetOutputTensorProperties(&ctx->output.properties, ctx->model, 0);
  if (ret != 0) {
    gesture_fail(ctx, ret, "hbDNNGetOutputTensorProperties");
    return ctx;
  }
  const hbDNNTensorShape *shape = &ctx->input.properties.validShape;
  if (ctx->input.properties.tensorType != HB_DNN_TENSOR_TYPE_S8 ||
      shape->numDimensions != 4 || shape->dimensionSize[0] != 1 ||
      shape->dimensionSize[1] != GESTURE_SEQUENCE_LENGTH ||
      shape->dimensionSize[2] != GESTURE_LANDMARKS ||
      shape->dimensionSize[3] != GESTURE_CHANNELS) {
    gesture_fail(ctx, -1, "unexpected gesture input tensor");
    return ctx;
  }
  if (ctx->output.properties.validShape.dimensionSize[3] != GESTURE_OUTPUTS) {
    gesture_fail(ctx, -1, "unexpected gesture output tensor");
    return ctx;
  }
  ret = hbSysAllocCachedMem(
      &ctx->input.sysMem[0], (uint32_t)ctx->input.properties.alignedByteSize);
  if (ret != 0) {
    gesture_fail(ctx, ret, "hbSysAllocCachedMem(input)");
    return ctx;
  }
  ret = hbSysAllocCachedMem(
      &ctx->output.sysMem[0], (uint32_t)ctx->output.properties.alignedByteSize);
  if (ret != 0) {
    gesture_fail(ctx, ret, "hbSysAllocCachedMem(output)");
    gesture_release_tensors(ctx);
    return ctx;
  }
  ctx->ready = 1;
  snprintf(ctx->error, sizeof(ctx->error), "ok");
  return ctx;
}

const char *gesture_model_last_error(void *opaque) {
  GestureContext *ctx = (GestureContext *)opaque;
  return ctx == NULL ? "null gesture context" : ctx->error;
}

int gesture_model_ready(void *opaque) {
  GestureContext *ctx = (GestureContext *)opaque;
  return ctx != NULL && ctx->ready;
}

int gesture_model_infer(void *opaque, const float *features, float *logits23) {
  GestureContext *ctx = (GestureContext *)opaque;
  if (ctx == NULL || !ctx->ready || features == NULL || logits23 == NULL) {
    return gesture_fail(ctx, -1, "invalid argument");
  }

  int8_t *dst = (int8_t *)ctx->input.sysMem[0].virAddr;
  memset(dst, 0, (size_t)ctx->input.sysMem[0].memSize);
  for (int frame = 0; frame < GESTURE_SEQUENCE_LENGTH; ++frame) {
    for (int point = 0; point < GESTURE_LANDMARKS; ++point) {
      for (int channel = 0; channel < GESTURE_CHANNELS; ++channel) {
        int shift = 7;
        if (ctx->input.properties.shift.shiftData != NULL &&
            channel < ctx->input.properties.shift.shiftLen) {
          shift = ctx->input.properties.shift.shiftData[channel];
        }
        int source = (frame * GESTURE_LANDMARKS + point) *
                         GESTURE_CHANNELS +
                     channel;
        int quantized = (int)floorf(ldexpf(features[source], shift));
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        int offset = frame * ctx->input.properties.stride[1] +
                     point * ctx->input.properties.stride[2] +
                     channel * ctx->input.properties.stride[3];
        dst[offset] = (int8_t)quantized;
      }
    }
  }
  int ret = hbSysFlushMem(&ctx->input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  if (ret != 0) {
    return gesture_fail(ctx, ret, "hbSysFlushMem(input)");
  }

  hbDNNInferCtrlParam ctrl;
  memset(&ctrl, 0, sizeof(ctrl));
  ctrl.bpuCoreId = HB_BPU_CORE_ANY;
  ctrl.dspCoreId = HB_DSP_CORE_ANY;
  ctrl.priority = HB_DNN_PRIORITY_LOWEST;
  hbDNNTaskHandle_t task = NULL;
  hbDNNTensor *output_ptr = &ctx->output;
  ret = hbDNNInfer(&task, &output_ptr, &ctx->input, ctx->model, &ctrl);
  if (ret == 0) {
    ret = hbDNNWaitTaskDone(task, 1000);
  }
  if (task != NULL) {
    hbDNNReleaseTask(task);
  }
  if (ret != 0) {
    return gesture_fail(ctx, ret, "hbDNNInfer/hbDNNWaitTaskDone");
  }
  ret = hbSysFlushMem(&ctx->output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
  if (ret != 0) {
    return gesture_fail(ctx, ret, "hbSysFlushMem(output)");
  }

  if (ctx->output.properties.tensorType == HB_DNN_TENSOR_TYPE_S32) {
    const uint8_t *src = (const uint8_t *)ctx->output.sysMem[0].virAddr;
    for (int channel = 0; channel < GESTURE_OUTPUTS; ++channel) {
      int shift = 0;
      if (ctx->output.properties.shift.shiftData != NULL &&
          channel < ctx->output.properties.shift.shiftLen) {
        shift = ctx->output.properties.shift.shiftData[channel];
      }
      const int32_t *value = (const int32_t *)(
          src + channel * ctx->output.properties.stride[3]);
      logits23[channel] = ldexpf((float)(*value), -shift);
    }
  } else if (ctx->output.properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    const uint8_t *src = (const uint8_t *)ctx->output.sysMem[0].virAddr;
    for (int channel = 0; channel < GESTURE_OUTPUTS; ++channel) {
      const float *value = (const float *)(
          src + channel * ctx->output.properties.stride[3]);
      logits23[channel] = *value;
    }
  } else {
    return gesture_fail(ctx, -1, "unsupported gesture output tensor");
  }
  snprintf(ctx->error, sizeof(ctx->error), "ok");
  return 0;
}

void gesture_model_destroy(void *opaque) {
  GestureContext *ctx = (GestureContext *)opaque;
  if (ctx == NULL) return;
  gesture_release_tensors(ctx);
  if (ctx->packed != NULL) {
    hbDNNRelease(ctx->packed);
  }
  free(ctx);
}
