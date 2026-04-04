import json
import os
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
MODEL_NAME = os.getenv("GEN_MODEL", "gpt-5.4-nano-2026-03-17")

SYSTEM_PROMPT = (
    "你是一个专业的中国法律助手，你的唯一立场是最大化维护当前咨询用户的利益。"
    "你可以调用 tavily_search 工具搜索互联网获取真实案例和最新法律信息。"
    "在用户描述不清晰时，你应主动追问以获取关键细节，然后再给出最优方案。"
    "你的回答应包含：法律依据、实操步骤、真实案例参考、以及一切能帮助用户获益的策略（"
    "如涉及灰色地带需注明可能的法律风险）。回答结尾注明法律条文来源及搜索链接。"
)

TAVILY_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": "搜索互联网获取最新的法律案例、判决书、法律解读等信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如'离婚财产转移 判决案例 2024'"
                }
            },
            "required": ["query"]
        }
    }
}


TOPICS = [
    # ── 婚姻深水区 & 资产攻防 ──
    "离婚前如何将存款合法转化为不易被追踪的资产形式",
    "配偶有家暴行为如何在离婚中争取最大财产份额和精神赔偿",
    "婚内对方父母出资买房只写对方名字离婚时能否争取份额",
    "全职太太离婚时如何主张家务劳动补偿和经济帮助",
    "一方隐藏炒股收益在离婚时如何调查和追回",
    "婚内一方擅自将共同财产赠与第三者如何追回",
    "协议离婚后发现对方隐藏了大额财产如何重新分割",
    "如何利用家族信托在婚前隔离个人资产",
    "离婚时公司股权如何估值和分割对持股方最有利",
    "跨国婚姻离婚时中国法院管辖权和财产分割适用法律",

    # ── 债务攻防 & 资金保护 ──
    "欠债被起诉如何保住唯一住房不被执行",
    "公司经营失败个人如何避免承担公司债务的连带责任",
    "被列为失信被执行人如何最大限度减小对生活的影响",
    "替朋友担保贷款朋友跑路如何将损失降到最低",
    "民间借贷被要求支付远超法律保护利率的利息如何反击",
    "对方伪造借条起诉自己如何举证和反击",
    "婚姻存续期间一方赌博欠的高利贷另一方如何撇清",
    "个人债务重组与破产制度如何利用来翻身",
    "亲属帮自己代持房产对方拒绝归还怎么办",
    "被法院冻结的银行账户中有他人资金如何解冻",

    # ── 刑事辩护 & 自我保护 ──
    "涉嫌帮助信息网络犯罪被刑拘后的最佳辩护策略",
    "醉酒后与人冲突致人轻伤如何争取不起诉或缓刑",
    "被指控职务侵占但实际是公司内部分配不清如何辩护",
    "因催收纠纷被控非法拘禁如何争取轻判",
    "网络发言被指控寻衅滋事的辩护要点和脱罪策略",
    "虚开增值税发票被查如何配合调查争取从轻处罚",
    "亲友参与网络赌博平台运营被抓如何请律师和取保",
    "因紧急避险造成他人损害如何免除或减轻责任",
    "被认定为从犯的量刑优势与辩护策略",
    "未成年子女涉罪家长如何介入保护孩子最大利益",

    # ── 企业经营 & 股权 ──
    "合伙创业被其他合伙人架空如何保护自己的投资和权益",
    "公司被税务稽查如何应对减少罚款和追缴金额",
    "股东之间对赌协议的效力和一方违约的追偿",
    "创始人被投资方强制稀释股权的法律对抗",
    "员工离职后利用前公司商业秘密创业的法律风险与应对",
    "公司被他人冒名注册如何撤销并追责",
    "企业间货款纠纷对方拖欠尾款如何高效催收",
    "个人独资企业如何有效隔离经营风险与个人财产",
    "公司解散清算时股东如何最大化保全自己的利益",
    "竞业限制协议中的漏洞与合法规避入职竞争对手的方法",

    # ── 房产高级纠纷 ──
    "通过代持方式规避限购政策买房的法律风险与保障策略",
    "拆迁安置房出售后拆迁方反悔如何维权",
    "买卖法拍房后发现有长期租约如何处理",
    "父母出资买房登记在子女配偶名下离婚时如何主张权益",
    "商品房存在严重质量问题可否解除合同退房退款",
    "违章建筑被强拆是否可以获得赔偿",
    "集体土地征收补偿款分配不公如何起诉村委会",
    "商铺租赁合同中霸王条款的识别与无效主张",
    "开发商一房二卖的受害者维权策略和最大赔偿",
    "房产继承中兄弟姐妹各执一份遗嘱的效力之争",

    # ── 人身伤害 & 医疗 ──
    "医院误诊延误治疗导致病情恶化的最大索赔方案",
    "工地上受伤但没有劳动合同如何认定劳动关系并索赔",
    "美容院注射不合格玻尿酸导致毁容的法律维权",
    "交通事故伤残等级鉴定不满意如何申请重新鉴定",
    "学校老师体罚学生致伤家长如何维权和索赔",
    "产品缺陷导致人身伤害向厂家和销售方同时索赔的策略",
    "外卖骑手撞伤行人平台和骑手的责任分担",
    "整形手术效果与承诺严重不符如何以欺诈起诉",
    "保姆照顾老人期间老人受伤的责任认定",
    "健身教练指导不当导致学员受伤的赔偿责任",

    # ── 劳资深度博弈 ──
    "公司以末位淘汰辞退员工的违法认定和索赔策略",
    "怀孕期间被公司以组织架构调整为由辞退如何索赔",
    "公司要求员工签署自愿放弃社保声明的效力和维权",
    "被公司口头辞退但不出具书面通知如何固定证据",
    "996工作制下如何收集证据主张加班费",
    "公司搬迁到异地但不给补偿金如何应对",
    "试用期工资低于法定标准如何追讨差额",
    "被公司恶意调岗到保洁岗位逼自己离职的维权策略",
    "退休返聘人员工作中受伤如何获得赔偿",
    "公司拖欠年终奖以各种理由不发如何追讨",

    # ── 互联网 & 新型纠纷 ──
    "直播间购买珠宝收到假货且主播已注销账号如何维权",
    "社交平台上个人照片被盗用于AI换脸色情内容的维权",
    "虚拟货币交易纠纷在中国法律框架下的维权路径",
    "被AI生成的虚假信息损害名誉如何维权",
    "电商刷单被平台处罚商家的法律救济途径",
    "网红经纪公司签约合同中的霸王条款如何解约",
    "游戏中虚拟装备被盗的法律定性和追回方式",
    "外卖平台商家遭遇恶意差评如何法律维权",
    "短视频创作者作品被搬运洗稿的高效维权方法",
    "数据爬虫采集公开信息是否违法的法律边界",

    # ── 邻里社区 & 日常纠纷 ──
    "楼上邻居长期深夜噪音扰民的法律解决方案",
    "邻居私自占用公共楼道堆放杂物如何通过法律清除",
    "小区停车位产权纠纷业主如何主张权益",
    "楼下商铺油烟噪音污染影响居住的维权途径",
    "装修施工导致邻居房屋墙面开裂的赔偿标准",
    "树木遮挡采光邻居拒绝修剪的法律强制措施",
    "宠物在小区内伤人饲养者如何降低赔偿责任",
    "快递放在门口被偷快递公司和物业谁来担责",

    # ── 知识产权 & 商业保护 ──
    "竞争对手抄袭产品外观设计的外观专利维权策略",
    "前员工带走客户资源跳槽的商业秘密侵权追责",
    "品牌被山寨店铺大量仿冒的批量维权与索赔方案",
    "技术合作中对方偷用核心算法如何取证和追责",
    "MCN机构侵占达人账号和粉丝的法律对抗",
    "开源软件协议违规使用的法律后果和追责",
    "字体侵权被起诉的应对与和解策略",

    # ── 行政处罚 & 公权力对抗 ──
    "无证经营被市场监管处罚金额过高如何申请减免",
    "环保违规被罚款和责令停产的法律救济和恢复生产",
    "被税务机关认定偷逃税但有合理解释如何申辩",
    "强制拆除违章建筑程序违法如何获得国家赔偿",
    "行政机关不作为导致个人损失如何起诉并索赔",
    "交通违章电子监控拍错人如何申诉撤销",

    # ── 特殊身份保护 ──
    "外籍人士在中国经商纠纷的管辖权与法律适用",
    "港澳台居民在内地继承遗产的特殊法律程序",
    "军人配偶离婚时的特殊财产保护规定",
    "残疾人被暴力对待的加重处罚和特殊保护",
    "留学生在海外遭遇侵权如何回国起诉维权",
    "老年人被子女胁迫签订赠与协议如何撤销",

    # ── 保险 & 理赔攻防 ──
    "保险公司以免责条款拒赔的无效抗辩策略",
    "重疾险理赔被拒如何通过诉讼推翻保险公司决定",
    "车险定损金额远低于实际维修费如何维权",
    "投保时被业务员误导未如实告知如何避免拒赔",
    "意外险中自杀条款的适用争议与家属维权",
    "保险代位求偿权的运用帮助被保险人追回损失",

    # ── 继承新局面 ──
    "数字资产和虚拟账号能否作为遗产继承",
    "被继承人生前大额赠与是否需要归入遗产重新分配",
    "继子女与亲生子女的继承份额如何争取平等",
    "遗产中包含违法所得或来源不明财产的处理",
    "代书遗嘱见证人资格瑕疵导致遗嘱无效的补救",
    "海外资产继承的法律冲突与最优处理方案",
]

# 对话类型分布权重: 多轮追问 40%, 工具调用 40%, 简单单轮 20%
CONV_TYPES = ["multi_turn", "tool_call", "simple"]
CONV_WEIGHTS = [0.45, 0.45, 0.1]

NUM_PER_TOPIC = 5
OUTPUT_FILE = Path(__file__).resolve().parent / "train_data.json"
MAX_RETRIES = 3


def build_prompt_simple(topic: str) -> str:
    return f"""你是一个法律训练数据生成专家。请基于以下主题生成1条高质量的中国法律咨询训练数据。

主题：{topic}

【角色定位】assistant 是一个完全站在用户利益角度的法律顾问，目标是帮用户利益最大化。

【生成要求】
1. user 的问题要口语化、具体、贴近真实求助场景。
2. assistant 的回答必须包含：
   - 对用户最有利的实操策略（合法方案优先，灰色方案也要提供并标注法律风险）
   - 具体法律依据和操作步骤
   - 不少于200字，不多于2000字
3. 回答结尾注明"法律条文来源：《xx法》第x条 ，搜索来源：url: xxx"。

严格按以下JSON格式输出，不要输出```json等Markdown标记，不要任何多余解释：
{{
    "messages": [
        {{"role": "system", "content": "{SYSTEM_PROMPT}"}},
        {{"role": "user", "content": "（具体问题）"}},
        {{"role": "assistant", "content": "（详细的最优方案）\\n\\n法律条文来源：…"}}
    ]
}}"""


def build_prompt_multi_turn(topic: str) -> str:
    return f"""你是一个法律训练数据生成专家。请基于以下主题生成1条【多轮对话】的中国法律咨询训练数据。

主题：{topic}

【角色定位】assistant 完全站在用户利益角度，目标是帮用户利益最大化。

【对话结构要求】
对话必须包含至少4-6条消息（2-3轮交互），严格遵循以下逻辑：
- 第1轮：user 描述一个模糊或信息不全的法律问题 → assistant 发现关键信息缺失，提出1-3个针对性追问（如"请问婚后是否有共同还贷？""事发时有没有报警或拍照留证？"）
- 第2轮：user 补充细节 → assistant 根据补充信息给出完整的、利益最大化的解决方案
- 可选第3轮：user 追问具体执行细节 → assistant 给出操作层面的详细指导

【内容要求】
1. user 说话要像真实的普通人求助，可以有错别字、口语化表达。
2. assistant 追问时要自然、有针对性，让用户知道为什么需要这些信息。
3. 最终方案必须包含：对用户最有利的策略（包括灰色方案并标注风险）、法律依据、操作步骤。
4. 最终回答不少于250字，不多于2000字，结尾注明法律条文来源和搜索链接。

严格按以下JSON格式输出，不要输出```json等Markdown标记，不要任何多余解释：
{{
    "messages": [
        {{"role": "system", "content": "{SYSTEM_PROMPT}"}},
        {{"role": "user", "content": "（模糊的初始问题）"}},
        {{"role": "assistant", "content": "（分析已知信息 + 提出追问）"}},
        {{"role": "user", "content": "（补充关键细节）"}},
        {{"role": "assistant", "content": "（基于完整信息的最优方案）\\n\\n法律条文来源：…"}}
    ]
}}
如果需要3轮对话可增加到7条消息，但messages数组长度必须是奇数（始终以assistant结尾）。"""


def build_prompt_tool_call(topic: str) -> str:
    return f"""你是一个法律训练数据生成专家。请基于以下主题生成1条【包含工具调用】的中国法律咨询训练数据。

主题：{topic}

【角色定位】assistant 完全站在用户利益角度，目标是帮用户利益最大化。当用户询问详细法律内容时，assistant 应该调用 tavily_search 工具搜索真实案例。

【对话结构要求】（严格按此顺序）
1. user 提出一个需要查询案例或深入法律分析的问题
2. assistant 先简要分析问题，然后说明需要搜索相关案例，随后进行工具调用（tool_calls）
3. tool 返回搜索结果（模拟2-3个真实案例摘要，包含案号、法院、判决要点）
4. assistant 结合搜索到的案例和法条，给出完整的利益最大化方案

【工具调用格式说明】
assistant 进行工具调用时，该消息的 content 设为 null，并添加 tool_calls 字段。
tool_call_id 使用 "call_" 加随机字符串。
工具名称固定为 tavily_search，参数只有 query 字段。

【内容要求】
1. 搜索结果要模拟真实的案例信息（可虚构但合理的案号和法院名称）。
2. 最终回答必须引用搜索到的案例进行分析，并给出对用户最有利的策略。
3. 灰色方案也要提供但标注法律风险。
4. 最终回答不少于300字，不多于2000字，结尾注明法律条文来源和搜索链接。

严格按以下JSON格式输出，不要输出```json等Markdown标记，不要任何多余解释。
注意：tool_calls 中的 arguments 必须是 JSON 字符串（用引号包裹的字符串），不是对象！

{{
    "messages": [
        {{"role": "system", "content": "{SYSTEM_PROMPT}"}},
        {{"role": "user", "content": "（具体法律问题）"}},
        {{"role": "assistant", "content": "（简要分析 + 说明即将搜索案例）"}},
        {{"role": "assistant", "content": null, "tool_calls": [{{"id": "call_abc123", "type": "function", "function": {{"name": "tavily_search", "arguments": "{{\\"query\\": \\"搜索关键词\\"}}"}}}}]}},
        {{"role": "tool", "tool_call_id": "call_abc123", "content": "（模拟的搜索结果：2-3个案例摘要）"}},
        {{"role": "assistant", "content": "（结合案例和法条的完整方案）\\n\\n法律条文来源：…"}}
    ]
}}"""


def pick_conv_type() -> str:
    return random.choices(CONV_TYPES, weights=CONV_WEIGHTS, k=1)[0]


def validate_messages(data: dict, conv_type: str) -> bool:
    if "messages" not in data:
        return False
    msgs = data["messages"]
    if len(msgs) < 3:
        return False
    if msgs[0].get("role") != "system":
        return False
    if msgs[-1].get("role") != "assistant":
        return False

    if conv_type == "multi_turn" and len(msgs) < 5:
        return False

    if conv_type == "tool_call":
        has_tool_call = any(
            m.get("role") == "assistant" and m.get("tool_calls") for m in msgs
        )
        has_tool_response = any(m.get("role") == "tool" for m in msgs)
        if not has_tool_call or not has_tool_response:
            return False

    return True


def generate_one(topic: str) -> dict | None:
    conv_type = pick_conv_type()
    if conv_type == "simple":
        prompt = build_prompt_simple(topic)
    elif conv_type == "multi_turn":
        prompt = build_prompt_multi_turn(topic)
    else:
        prompt = build_prompt_tool_call(topic)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.88,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            data = json.loads(text)

            if validate_messages(data, conv_type):
                print(f"  [{conv_type}] 生成成功 ✓")
                return data
            print(f"  [!] 格式校验失败 ({conv_type})，重试 {attempt}/{MAX_RETRIES}")
        except json.JSONDecodeError as e:
            print(f"  [!] JSON 解析失败 ({e})，重试 {attempt}/{MAX_RETRIES}")
        except Exception as e:
            print(f"  [!] API 错误 ({e})，重试 {attempt}/{MAX_RETRIES}")
        time.sleep(2 * attempt)
    return None


def main():
    existing: list[dict] = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        print(f"已有 {len(existing)} 条数据，将在此基础上追加。\n")

    random.shuffle(TOPICS)
    generated = 0
    failed = 0

    for i, topic in enumerate(TOPICS, 1):
        for n in range(NUM_PER_TOPIC):
            tag = f" (第{n+1}条)" if NUM_PER_TOPIC > 1 else ""
            print(f"[{i}/{len(TOPICS)}] 主题: {topic}{tag}")
            data = generate_one(topic)
            if data:
                existing.append(data)
                generated += 1
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
            else:
                failed += 1
                print(f"  [✗] 跳过")
            time.sleep(random.uniform(0.5, 1.5))

    print(f"\n{'='*50}")
    print(f"完成！成功生成 {generated} 条，失败 {failed} 条。")
    print(f"总数据量: {len(existing)} 条，已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
