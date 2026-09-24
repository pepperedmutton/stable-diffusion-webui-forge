def register(options_templates, options_section, OptionInfo):
    options_templates.update(options_section((None, "Forge Hidden options"), {
        "forge_unet_storage_dtype": OptionInfo('Automatic'),
        "forge_inference_memory": OptionInfo(1024),
        "forge_async_loading": OptionInfo('Queue'),
        "forge_pin_shared_memory": OptionInfo('CPU'),
        "forge_preset": OptionInfo('qwen'),
        "forge_additional_modules": OptionInfo([]),
        "forge_checkpoint_qwen": OptionInfo('Qwen-Image-2.1'),
        "forge_additional_modules_qwen": OptionInfo([]),
        "forge_unet_storage_dtype_qwen": OptionInfo('Automatic'),
        "forge_qwen21_profile_version": OptionInfo(0),
        "forge_checkpoint_krea": OptionInfo('krea2Cocoamixzero_v10.safetensors'),
        "forge_additional_modules_krea": OptionInfo([]),
        "forge_unet_storage_dtype_krea": OptionInfo('Automatic'),
        "forge_checkpoint_xl": OptionInfo('miaomiaoHarem_v20.safetensors'),
        "forge_additional_modules_xl": OptionInfo([]),
        "forge_additional_modules_xl_configured": OptionInfo(False),
        "forge_checkpoint_anima": OptionInfo('miaomiaoHarem_anima15.safetensors'),
        "forge_additional_modules_anima": OptionInfo([]),
    }))
    options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
        "forge_canvas_plain": OptionInfo(False, "ForgeCanvas: use plain background").needs_reload_ui(),
        "forge_canvas_toolbar_always": OptionInfo(False, "ForgeCanvas: toolbar always visible").needs_reload_ui(),
    }))
