import torch
import sys

def verificar_sistema():
    print("🕵️ DIAGNÓSTICO DE SISTEMA PARA IA")
    print("="*40)
    
    # 1. Versión de Python
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # 2. Versión de PyTorch
    print(f"🔥 PyTorch: {torch.__version__}")
    
    # 3. Detección de CUDA
    cuda_disponible = torch.cuda.is_available()
    print(f"✅ CUDA Disponible: {'SÍ' if cuda_disponible else 'NO ❌'}")
    
    if cuda_disponible:
        count = torch.cuda.device_count()
        print(f"🔢 Cantidad de GPUs: {count}")
        for i in range(count):
            print(f"   -> GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Prueba de memoria
        try:
            x = torch.rand(5, 3).cuda()
            print("\n✅ Prueba de Tensor en VRAM: ÉXITO (La GPU está respondiendo)")
        except Exception as e:
            print(f"\n❌ Error al escribir en GPU: {e}")
    else:
        print("\n⚠️ ADVERTENCIA CRÍTICA:")
        print("   Tu PyTorch no está detectando la GPU.")
        print("   Posible causa: Tienes instalada la versión 'cpu-only'.")
        print("   Solución: Ejecuta el comando de reinstalación con --index-url cu121.")

if __name__ == "__main__":
    verificar_sistema()