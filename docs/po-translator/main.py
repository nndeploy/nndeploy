import polib
import json
from json import JSONDecodeError
from tqdm import tqdm
from openai import OpenAI
from config import *
from rate_limiter import RateLimiter


class AiTranslator:
    def __init__(self, config, api_key):
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=config['base_url'],
        )
        self.prompt = config['prompt']
        self.rate_limiter = RateLimiter(config.get('rpm', 60), 60)

    def translate_text(self, text):
        @self.rate_limiter
        def limited_chat(msgs):
            return self.client.chat.completions.create(
                model=self.config['model'],
                messages=msgs,
                stream=False,
                response_format={"type": "json_object"},
                temperature=self.config.get('temperature', 1)
            )

        for attempt in range(MAX_RETRIES):
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ]
            completion_contents = []
            try:
                while True:
                    completion = limited_chat(messages)
                    messages.append(completion.choices[0].message)
                    completion_contents.append(completion.choices[0].message.content)
                    if completion.choices[0].finish_reason != "length":
                        break

                return ''.join(completion_contents)
            except Exception as e:
                logger.error(f"尝试 {attempt + 1} 次失败：{e}")
                if attempt >= MAX_RETRIES - 1:
                    logger.error("已达到最大重试次数。返回空。")

        return ''


class PoWalkTranslator:
    def __init__(self, translator: AiTranslator, src: str, dest: str = None):
        self.translator = translator
        self.src = src
        self.dest = dest

    def run(self):
        for root, dirs, files in os.walk(self.src):
            logger.info(f"正在翻译目录{root}")
            for file in files:
                if file.endswith(".po"):
                    file_path = os.path.join(root, file)
                    new_file_path = file_path.replace(self.src, self.dest) if self.dest else file_path
                    self.translate_po_file(file_path, new_file_path)

    def translate_po_file(self, file_path, new_file_path=None):
        logger.info(f"正在翻译: {file_path}")

        po = polib.pofile(file_path)
        untranslated_entries = po.untranslated_entries()

        if not untranslated_entries:
            logger.info("没有需要翻译的条目，跳过翻译。")
            return

        # 处理批次
        batches = []
        current_batch = []
        current_length = 0

        for entry in untranslated_entries:
            if current_length + len(entry.msgid) > BATCH_MAX_CHARTS:
                batches.append(current_batch)
                current_batch = []
                current_length = 0
            current_batch.append(entry)
            current_length += len(entry.msgid)

        if current_batch:
            batches.append(current_batch)

        translation_success = True  # 用于标记翻译是否成功

        # 按批次翻译
        for batch in tqdm(batches, desc="翻译进度"):
            batch_map = {str(index): entry.msgid for index, entry in enumerate(batch)}
            content = json.dumps(batch_map)
            translated_content = self.translator.translate_text(content)
            try:
                translated_batch_map = json.loads(translated_content)
                if not isinstance(translated_batch_map, dict):
                    raise ValueError("翻译结果不是有效的字典格式")
            except (JSONDecodeError, ValueError) as e:
                logger.error(f"解析翻译结果失败: {e}")
                logger.error("可能是翻译的条目过多，丢失该部分翻译，请尝试修改 BATCH_MAX_CHARTS 配置重新运行")
                logger.error(f"出错的文件: {file_path}")
                translation_success = False
                break  # 跳出循环，不再继续翻译当前文件

            if not translated_content:
                logger.info(f"翻译失败原文: {content}")
                logger.warning("警告: 批次翻译失败，保持原有的空翻译")
                translation_success = False
                break  # 跳出循环，不再继续翻译当前文件

            # 更新翻译
            for key, value in translated_batch_map.items():
                msgid, msgstr = batch_map[key], value
                entry = po.find(msgid)
                if entry:
                    if entry.msgid_plural:
                        entry.msgstr_plural['0'] = msgstr
                        entry.msgstr_plural['1'] = msgstr
                    else:
                        entry.msgstr = msgstr

        if translation_success:
            # 保存翻译文件
            to_file_path = new_file_path or file_path
            self.ensure_directory_exists(to_file_path)
            po.save(to_file_path)
            logger.info(f"已保存翻译后的文件到: {to_file_path}\n")
        else:
            logger.warning(f"由于错误，跳过保存文件: {file_path}")


    @staticmethod
    def ensure_directory_exists(file_path):
        """
        确保文件路径的目录存在，如果不存在则创建。
        :param file_path: 目标文件的完整路径
        """
        directory = os.path.dirname(file_path)
        if directory:  # 如果路径中包含目录
            os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    translator = AiTranslator(MODEL_CONFIG, API_KEY)
    po_walk_translator = PoWalkTranslator(translator, FROM_DIR, TO_DIR)
    po_walk_translator.run()
