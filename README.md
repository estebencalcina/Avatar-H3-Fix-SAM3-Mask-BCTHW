# ComfyUI-AvatarH3-SAM3-MaskFix

Nodo de adaptacion de mascara de video SAM3 para latentes combinados de video/audio MiniMax H3 usados por SelfLift Avatar.

## Nodo incluido

- **Avatar H3 · Fix SAM3 Mask BCTHW** (`AvatarH3FixMask`)
- Entrada: `latent` (`LATENT`); salida: `latent` (`LATENT`).
- Convierte una mascara de video `[T,1,H,W]` en `[1,1,T,H,W]`, manteniendo la mascara de audio.
- Solo admite latentes anidados de video/audio con las formas esperadas de MiniMax H3 y mascara `noise_mask` anidada.

## Instalacion con ZIP

1. Descomprime el ZIP en `ComfyUI/custom_nodes/`. Debe quedar `ComfyUI/custom_nodes/ComfyUI-AvatarH3-SAM3-MaskFix/__init__.py`.
2. Si anteriormente instalaste `avatar_h3_mask_fix.py` directamente en `custom_nodes`, quitalo o muevelo fuera de ese directorio para no registrar el mismo nodo dos veces.
3. Reinicia ComfyUI. No hay `pip install` adicional: utiliza PyTorch y los modulos incorporados en ComfyUI.
4. Abre tu workflow y confirma que aparece `Avatar H3 · Fix SAM3 Mask BCTHW`.

## Publicarlo en GitHub

1. Crea un repositorio publico con el nombre `ComfyUI-AvatarH3-SAM3-MaskFix`.
2. Sube a la raiz del repositorio `__init__.py`, `mask_fix.py` y `README.md` (extrae el ZIP antes de subir los archivos).
3. Para instalar desde GitHub, clona el repositorio dentro de `ComfyUI/custom_nodes/` y reinicia ComfyUI.

## Conexion en ComfyUI

`Latente MiniMax H3 con noise_mask de SAM3` → `AvatarH3FixMask` → `SelfLiftAvatarH3Sampler.latent_image`

## Aviso

Este es un adaptador de **forma de tensor**, no un sistema completo de segmentacion. No corrige errores de seguimiento SAM3, sincronizacion de fotogramas ni discrepancias entre los latentes de audio/video. No se ha validado la ejecucion en GPU en este entorno.
