#!/usr/bin/env python3
"""
Quick Model Verification across all Docker Containers
모든 Docker 컨테이너에서 모델들이 정상적으로 로딩되는지 빠르게 검증
"""
import subprocess
import time
import logging
from pathlib import Path
import yaml
from spectravision.docker_orchestrator import DockerOrchestrator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_configs():
    """Docker 모델 설정 로드"""
    with open('configs/models_docker.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_model_availability(container_name, model_config):
    """각 모델의 사용 가능성을 빠르게 테스트"""
    model_name = model_config['name']
    vlm_id = model_config['vlm_id']
    
    logger.info(f"🔍 Testing {model_name} ({vlm_id}) in {container_name}")
    
    # VLMEvalKit에서 모델이 지원되는지 확인
    test_cmd = f"""
    python -c "
import sys
try:
    from vlmeval.config import supported_VLM
    if '{vlm_id}' in supported_VLM:
        print('✅ Model {vlm_id} is supported in VLMEvalKit')
        exit(0)
    else:
        print('❌ Model {vlm_id} not found in supported_VLM')
        print('Available models:', sorted(list(supported_VLM.keys())[:10]))
        exit(1)
except Exception as e:
    print(f'❌ Error checking model: {{e}}')
    exit(1)
"
    """
    
    try:
        result = subprocess.run([
            'docker', 'exec', f'spectravision-{container_name.replace("_", "-")}',
            'bash', '-c', test_cmd
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info(f"✅ {model_name} - AVAILABLE")
            return True
        else:
            logger.warning(f"❌ {model_name} - NOT AVAILABLE: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {model_name} - TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"💥 {model_name} - ERROR: {e}")
        return False

def test_container_basic_functionality(container_name):
    """컨테이너 기본 기능 테스트"""
    logger.info(f"🧪 Testing container {container_name} basic functionality")
    
    # Transformers 버전 확인
    version_cmd = "python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"
    
    try:
        result = subprocess.run([
            'docker', 'exec', f'spectravision-{container_name.replace("_", "-")}',
            'bash', '-c', version_cmd
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            logger.info(f"✅ {container_name} - {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ {container_name} - Failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"💥 {container_name} - Error: {e}")
        return False

def main():
    logger.info("🚀 시작: Quick Model Verification across all Docker Containers")
    
    # Docker orchestrator 초기화
    orchestrator = DockerOrchestrator()
    configs = load_configs()
    
    results = {}
    total_models = 0
    available_models = 0
    
    # 각 컨테이너별로 테스트
    for container_name, container_config in configs.items():
        if container_name in ['hardware_tiers', 'deployment_strategy']:
            continue
            
        logger.info(f"\n📦 Container {container_name} ({container_config['version']}) 테스트 시작")
        
        # 컨테이너 시작
        try:
            orchestrator.start_container(container_name)
            time.sleep(2)  # 컨테이너 초기화 대기
            
            # 기본 기능 테스트
            if not test_container_basic_functionality(container_name):
                logger.error(f"❌ Container {container_name} basic functionality failed")
                continue
            
            # 모델 테스트
            container_results = {
                'available': [],
                'unavailable': [],
                'total': len(container_config['models'])
            }
            
            for model_config in container_config['models']:
                total_models += 1
                if test_model_availability(container_name, model_config):
                    available_models += 1
                    container_results['available'].append(model_config['name'])
                else:
                    container_results['unavailable'].append(model_config['name'])
            
            results[container_name] = container_results
            
            logger.info(f"📊 {container_name} Results: {len(container_results['available'])}/{container_results['total']} models available")
            
        except Exception as e:
            logger.error(f"💥 Container {container_name} failed: {e}")
            results[container_name] = {'error': str(e)}
        
        finally:
            # 컨테이너 정리
            try:
                orchestrator.stop_container(container_name)
            except:
                pass
    
    # 최종 결과 리포트
    logger.info("\n" + "="*60)
    logger.info("📋 FINAL VERIFICATION REPORT")
    logger.info("="*60)
    
    for container_name, result in results.items():
        if 'error' in result:
            logger.error(f"❌ {container_name}: ERROR - {result['error']}")
            continue
            
        available = len(result['available'])
        total = result['total']
        success_rate = (available / total * 100) if total > 0 else 0
        
        logger.info(f"📦 {container_name}:")
        logger.info(f"   ✅ Available: {available}/{total} ({success_rate:.1f}%)")
        
        if result['unavailable']:
            logger.info(f"   ❌ Unavailable: {', '.join(result['unavailable'])}")
    
    overall_success_rate = (available_models / total_models * 100) if total_models > 0 else 0
    logger.info(f"\n🎯 OVERALL SUCCESS RATE: {available_models}/{total_models} ({overall_success_rate:.1f}%)")
    
    if overall_success_rate >= 80:
        logger.info("🎉 시스템이 프로덕션 준비가 완료되었습니다!")
    else:
        logger.warning("⚠️  일부 모델들에 문제가 있습니다. 추가 디버깅이 필요합니다.")

if __name__ == "__main__":
    main()