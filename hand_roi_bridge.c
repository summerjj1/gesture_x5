#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>

#define HAND_OUTPUTS 4
#define HAND_LANDMARK_VALUES 63

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
