"""
Ecosystem tagging for ComfyUI Workflow Studio.
Deterministic tag assignment based on node type presence — no LLM required.
"""


# Anchor node types that strongly indicate an ecosystem.
# Ordered by specificity — more specific entries first.
ECOSYSTEM_ANCHORS = {
    # Video — model-specific
    'wan':         {'WanVideoSampler', 'WanVideoLoader', 'WanVideoEncode', 'WanVideoLora',
                    'WanVideoNAG', 'WanVideoBlockSwap', 'WanVideoEnhanceAVideo',
                    'CLIPTextEncodeWan', 'DaSiWa_Wan22'},
    'ltx':         {'LTXVLoader', 'LTXVSampler', 'LTXVScheduler', 'LTXVConditioning',
                    'LTXVImgToVideo', 'LTXVPreprocess', 'LTXVAddAudio',
                    'LTXVAudioVAEDecode', 'LTXVConcatAVLatent', 'CLIPTextEncodeLTXV'},
    'hunyuan_vid': {'HunyuanVideoSampler', 'HunyuanVideoLoader', 'HunyuanVideoDecode',
                    'CLIPTextEncodeHunyuan', 'HunyuanVideoVAELoader'},
    'mochi':       {'MochiWrapper', 'MochiSampler', 'MochiModelConfig'},
    'cogvideo':    {'CogVideoSampler', 'CogVideoLoader', 'CogVideoEncode'},
    'animatediff': {'AnimateDiffLoader', 'AnimateDiffSampler', 'ADE_AnimateDiffLoaderV1Gen2'},
    # Image — model-specific
    'nunchaku':    {'NunchakuFluxDiTLoader', 'NunchakuFluxLoraLoader',
                    'NunchakuTextEncoderLoaderV2', 'NunchakuFluxPuLIDApplyV2'},
    'zimage':      {'ZSamplerTurbo2', 'StyleStringInjector2', 'StylePromptEncoder2',
                    'ZImageAnalyzerSelectiveLoaderV2', 'ZEngineer', 'ZSamplerClass'},
    'flux':        {'FluxGuidance', 'ModelSamplingFlux', 'CLIPTextEncodeFlux',
                    'Flux2Scheduler', 'EmptyFlux2LatentImage', 'FluxKontextImageScale'},
    'sd3':         {'ModelSamplingSD3', 'CLIPTextEncodeSD3', 'EmptySD3LatentImage',
                    'TripleCLIPLoader'},
    'qwen_edit':   {'TextEncodeQwenImageEditPlus', 'QwenEditConfigPreparer',
                    'TextEncodeQwenImageEditPlusCustom_lrzjason', 'ModelSamplingAuraFlow'},
    'hunyuan_3d':  {'EmptyLatentHunyuan3Dv2', 'VAEDecodeHunyuan3D',
                    'Hunyuan3Dv2ConditioningMultiView', 'SaveGLB'},
    'training':    {'AnimaTrainingLauncher', 'AnimaModelDownloader', 'AnimaTrainingWizard',
                    'AnimaSDScriptsManager'},
    # Capability tags (can coexist with model tags)
    'video':       {'VHS_VideoCombine', 'VHS_LoadVideo', 'RIFE VFI', 'RIFEInterpolation',
                    'VideoToImages', 'SaveVideo', 'LoadVideo'},
    'audio':       {'ChatterBoxEngineNode', 'F5TTSEngineNode', 'UnifiedTTSSRTNode',
                    'CharacterVoicesNode', 'LoadAudio', 'SaveAudio', 'MMAudio',
                    'LTXVAddAudio', 'EmptyAudioLatent'},
    'inpainting':  {'InpaintModelConditioning', 'VAEEncodeForInpaint', 'LanPaintNode',
                    'DifferentialDiffusion', 'InpaintCropImproved'},
    'upscaling':   {'UltimateSDUpscale', 'UltimateSDUpscaleCustomSample',
                    'UltimateSDUpscaleNoUpscale', 'SeedVR2VideoUpscaler',
                    'ImageUpscaleWithModel'},
    'face':        {'FaceDetailer', 'FaceDetailerPipe', 'ReActorFaceSwap',
                    'NunchakuFluxPuLIDApplyV2', 'IPAdapterFaceID',
                    'InstantIDModelLoader', 'PulidModelLoader', 'ACE_Plus'},
    'controlnet':  {'ControlNetApplyAdvanced', 'ControlNetApplySD3',
                    'ControlNetLoader', 'AIO_Preprocessor',
                    'DepthAnythingV2Preprocessor', 'OpenposePreprocessor'},
    'captioning':  {'Florence2Run', 'DownloadAndLoadFlorence2Model',
                    'WD14Tagger', 'JoyCaptionAlpha', 'BLIPCaption',
                    'CLIPInterrogator', 'DeepDanbooru'},
    'segmentation':{'SAMLoader', 'SAMPredictor', 'GroundingDinoSAMSegment',
                    'DownloadAndLoadSAM2Model', 'SegmentAnything2',
                    'UltralyticsDetectorProvider', 'BboxDetectorSEGS'},
    'batch':       {'CR Prompt List', 'ImpactWildcardProcessor', 'LoadImageBatch',
                    'LoadImagesFromDirectory', 'VHS_LoadImages'},
    '3d':          {'SaveGLB', 'VoxelToMesh', 'EmptyLatentHunyuan3Dv2',
                    'Hunyuan3Dv2ConditioningMultiView'},
}

# Tags that imply "video" even if the direct video anchor isn't present
_VIDEO_IMPLIES = {'wan', 'ltx', 'hunyuan_vid', 'mochi', 'cogvideo', 'animatediff'}
# Tags that imply "image generation" as primary purpose
_IMAGE_GEN_IMPLIES = {'flux', 'nunchaku', 'zimage', 'sd3', 'qwen_edit'}


def tag_workflow(node_types: set) -> list[str]:
    """Return sorted list of ecosystem/capability tags for a workflow."""
    tags = []
    for tag, anchors in ECOSYSTEM_ANCHORS.items():
        if anchors & node_types:
            tags.append(tag)

    # Derived tags
    has_video_model = bool(set(tags) & _VIDEO_IMPLIES)
    has_sampler     = bool({'KSampler', 'KSamplerAdvanced', 'SamplerCustomAdvanced',
                            'WanVideoSampler', 'LTXVSampler', 'HunyuanVideoSampler'} & node_types)
    has_video_io    = bool({'VHS_VideoCombine', 'VHS_LoadVideo', 'SaveVideo'} & node_types)

    if has_video_model or (has_video_io and has_sampler):
        if 'video' not in tags:
            tags.append('video')
    if has_sampler and 'video' not in tags:
        if 'image_gen' not in tags:
            tags.append('image_gen')

    return sorted(set(tags))
