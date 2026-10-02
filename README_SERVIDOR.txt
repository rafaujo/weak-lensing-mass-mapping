PACOTE COM SMOKE VERIFICADO — código v6 preservado, 02/10/2026

STATUS ATUAL
O smoke planejado de um caso C0/R10/P0 está concluído. Os resultados da v5
validaram KS, Starlet, G+S e GLIMPSE; a v6 confirmou o MCALens com o script
do pacote e o NPZ dentro da nova pasta de saída. O relatório do servidor
registra o mesmo SHA256, tamanho e mtime do arquivo histórico antes/depois.
As métricas foram recalculadas do mapa enviado: diferença máxima de 1,21e-9
contra o CSV arquivado. Não é preciso repetir o servidor para esta correção.

Este pacote mantém exatamente o código testado na v6. Foram atualizados
somente licença, documentação/status e evidências compactas. Os mapas e dados brutos
enviados ficam no arquivo local de auditoria, fora do pacote de publicação.
Veja provenance/server_validation_v6/ASSESSMENT.txt e independent_audit.json.
O teste cobre um patch; não equivale a refazer os 480 casos ou instalar tudo
em ambiente novo. A MIT foi aprovada para as contribuições próprias de Rafael;
os componentes de terceiros mantêm suas licenças (veja LICENSE_SCOPE.md).
Repositório: https://github.com/rafaujo/weak-lensing-mass-mapping
Release v1.0.0 arquivado: https://doi.org/10.5281/zenodo.23102124
Os 200 arquivos do ZIP no Zenodo coincidem com o commit do release.
Para obter a versão exata arquivada, use git checkout v1.0.0 após o clone.

Os comandos abaixo ficam como instruções para futuras execuções; não são
uma nova tarefa de verificação exigida do autor.
O modo padrão executa quatro métodos; --with-glimpse inclui GLIMPSE.
--mcalens-only executa apenas a reconstrução MCALens. Essas duas opções
não podem ser combinadas. Cada chamada cria sua própria pasta de saída.

A v4 chamava MCALens no Python gob5 e não verificava sua cadeia de imports.
Isso é compatível com o erro informado: No module named 'modopt'.
A conversa “Revisão do artigo” confirma ambientes separados. A v5 mantém:

Etapa                              Ambiente
KS / Starlet / G+S                  /tatu/venv/gob5/bin/python
MCALens                            ~/venvs/mcalens/bin/python
Base do venv MCALens                ~/micromamba/envs/glimpse
Binário GLIMPSE (opcional)          ambiente micromamba glimpse

MCALens foi criado como venv --system-site-packages sobre glimpse. A tentativa
anterior de criar um ambiente micromamba chamado mcalens falhou no histórico;
por isso a v5 NÃO depende de “micromamba activate mcalens”.

COMO RODAR NOVAMENTE, SE DESEJADO
Obtenha uma cópia do repositório numa pasta nova:

  git clone https://github.com/rafaujo/weak-lensing-mass-mapping.git
  cd weak-lensing-mass-mapping
  bash run_server_smoke.sh --preflight-only
  bash run_server_smoke.sh

Não precisa alternar manualmente os ambientes entre as etapas.
Use o nó/alocação em que os ambientes científicos já funcionavam.

config_paths.server.toml contém os quatro caminhos de dados/código/saídas.
config_environments.server.toml contém os comandos dos ambientes. O MCALens usa:

  ~/bin/micromamba run -p ~/micromamba/envs/glimpse ~/venvs/mcalens/bin/python

O gob5 executa KS/Starlet/G+S; MCALens é um subprocesso separado, lê o mesmo
cache e devolve seu NPZ/relatório. Nenhum pacote é instalado, atualizado ou
copiado entre ambientes. Nada modifica o ambiente científico existente.
Se o Python principal mudou, ajuste gob5_python no TOML e WL_PYTHON no comando.
O executável micromamba, o prefixo glimpse e o Python MCALens também são
editáveis no TOML; não há caminhos científicos novos nos drivers.

PRÉ-CHECAGEM
--preflight-only confere hashes dos dados/prior, invariantes do código,
commit/patch do CosmoStat, e faz imports reais nos dois interpretadores.
No MCALens inclui modopt, seaborn e pycs.astro.wl.mass_mapping; não exige torch.
Registra sys.executable, versões, sys.prefix/base_prefix e origem dos módulos.
Falhas de importação impedem iniciar qualquer reconstrução.
Essa checagem importa bibliotecas, mas não importa os drivers de benchmark,
não chama os solvers nem gera mapas científicos.

As saídas ficam em work/server_smoke/<data-hora-id>/, exclusivo por execução:
  smoke_report.json         resultado consolidado e caminhos/ambientes
  imports_gob5.json/.log    imports efetivos no gob5
  imports_mcalens.json/.log imports efetivos no mcalens
  mcalens_worker.json       resultado do subprocesso MCALens
  mcalens.log               stdout/stderr do MCALens
  maps/                    reconstruções do único patch

Se um comando/ambiente não existir ou um import falhar, veja o .log/.json
correspondente. O erro e o interpretador ficam registrados. Não instale modopt
no gob5 para contornar um erro de seleção do ambiente.

O TESTE NUMÉRICO
Executa APENAS C0/R10/P0, 176 x 176, com as funções científicas preservadas:
KS sigma=1.5; Starlet 6000 iterações; G+S 6000 iterações, lambda=.05,
beta=.001; MCALens Nsigma=5, 100 iterações, quatro escalas.
O lote é um patch; o orçamento original de iterações permanece completo.
Não executa fusion, ADMM, Wiener, figuras, ou o benchmark dos 480 casos.

Reutiliza uma cópia da observação histórica:
  /u/gob5/Glimpse/test/fair_npz/C0_R10.npz
Valida a cópia, a verdade contra o FAIR bruto, cosmologia, sigma do ruído,
dimensões e finitude. O SHA256 histórico independente de gamma não foi fornecido.
Dados brutos e resultados antigos são apenas lidos, sem sobrescrever arquivos.

Se esse cache faltar, gera-se explicitamente com:
  bash run_server_smoke.sh --generate-input
Isso usa a função original, campo completo 1424 x 176 e seed=10, antes do recorte.
CPU/CUDA podem produzir ruídos diferentes; divergências exigem análise.

GLIMPSE OPCIONAL
O caso da v5 levou cerca de 14 minutos:
  bash run_server_smoke.sh --with-glimpse
Também repete os quatro métodos anteriores. O binário é executado com as
bibliotecas do ambiente glimpse; o preflight usa ldd nesse ambiente para
identificar bibliotecas ausentes. Catálogo/configuração usam as funções
preservadas do driver: lambda=3, niter=500, nrandom=1000, nreweights=5 e
GSL_RNG_SEED=1000. Não há recompilação.

SCRIPTS COMPLETOS (não executados automaticamente pelo smoke)
run_server_pipeline.sh seleciona o ambiente conforme o driver. Para conferir
um comando sem executá-lo:

  bash run_server_pipeline.sh --dry-run run_mcalens_test.py
  bash run_server_pipeline.sh --dry-run run_gaussian_starlet_test.py
  bash run_server_pipeline.sh --dry-run run_glimpse_test_final.py

Os três produtores MCALens e build_fair_validation_pk.py vão para mcalens;
os drivers em code/glimpse recebem o ambiente nativo glimpse; os demais
usam gob5. Remover --dry-run executa o script completo, com suas dependências
de arquivos e ordem de execução descritas no README. Para uma reprodução
completa, configure uma work_root dedicada; o smoke não cria todas as entradas.

STATUS/CÓDIGOS DE SAÍDA
0: preflight pronto OU smoke terminado dentro da tolerância de comparação.
1: dependência/import ausente, entrada inválida ou erro de execução.
2: terminou, mas alguma diferença numérica contra o CSV requer investigação.
A tolerância absoluta 1e-4 nas métricas NMSE/PCC/std_ratio é um diagnóstico
do smoke, não um hiperparâmetro científico nem critério histórico de seleção.

27 testes locais de suporte e 162 conferências das tabelas passaram. Os 25
scripts científicos continuam preservados. Incluímos teste de regressão que
simula a falta de modopt e comprova que ela bloqueia a pré-checagem, e um teste
real de subprocesso com driver fictício comprovando o isolamento do MCALens.
Os ambientes e as métricas dos cinco mapas foram conferidos nos resultados
da v5. O isolamento do MCALens, que falhou naquela versão, foi corrigido
e confirmado pelos resultados da v6. O código testado está preservado aqui.
