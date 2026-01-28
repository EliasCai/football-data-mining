import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import os

# 配置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OkoooScraper:
    def __init__(self):
        self.base_url = "https://www.okooo.com"
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': 'https://www.okooo.com/',
            'Connection': 'keep-alive',
        }
        self._prepare_session()

    def _prepare_session(self):
        """访问主页预热获取必要的Cookie"""
        try:
            logging.info("正在初始化会话，获取预热Cookie...")
            self.session.get(self.base_url, headers=self.headers, timeout=10)
            time.sleep(1)
        except Exception as e:
            logging.warning(f"预热访问失败（可能不影响后续请求）: {e}")

    def fetch_html(self, date):
        """请求目标页面"""
        target_url = f"{self.base_url}/livecenter/zucai/?mf=ToTo&date={date}"
        logging.info(f"正在请求日期数据: {date} URL: {target_url}")

        try:
            response = self.session.get(target_url, headers=self.headers, timeout=15)
            # if response.status_code == 200:
            #     response.encoding = 'utf-8'
            #     return response.text

            if response.status_code == 200:
            # 自动根据页面内容识别编码（比手动指定 utf-8 更稳健）
                response.encoding = response.apparent_encoding
                return response.text
            else:
                logging.error(f"请求失败，状态码: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"网络请求异常: {e}")
            return None

    def parse_html(self, html_content):
        """解析HTML内容并提取数据"""
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        # 查找所有含 matchid 的 tr
        match_rows = soup.find_all('tr', {'matchid': True})
        logging.info(f"成功定位到 {len(match_rows)} 场比赛数据")

        data_list = []
        for row in match_rows:
            try:
                # 提取各项字段
                match_id = row.get('matchid')

                # 赛事名称
                league_elem = row.find('a', class_='ssname')
                league = league_elem.text.strip() if league_elem else ''

                # 比赛时间 (取最后一个 graytx 类的 td)
                time_elems = row.find_all('td', class_='graytx')
                match_time = time_elems[-1].text.strip() if time_elems else ''

                # 队名与比分
                home_team = row.find('a', class_='ctrl_homename').text.strip() if row.find('a', class_='ctrl_homename') else ''
                away_team = row.find('a', class_='ctrl_awayname').text.strip() if row.find('a', class_='ctrl_awayname') else ''

                home_score = row.find('b', class_='ctrl_homescore').text.strip() if row.find('b', class_='ctrl_homescore') else ''
                away_score = row.find('b', class_='ctrl_awayscore').text.strip() if row.find('b', class_='ctrl_awayscore') else ''

                # SP值解析
                sp_elem = row.find('td', class_='blockbox ctrl_self_betopt')
                sp_values = sp_elem.find_all('span') if sp_elem else []
                sp_win = sp_values[0].text.strip() if len(sp_values) > 0 else ''
                sp_draw = sp_values[1].text.strip() if len(sp_values) > 1 else ''
                sp_lose = sp_values[2].text.strip() if len(sp_values) > 2 else ''

                # 分析链接
                analysis_td = row.find('td', class_='linebgdata')
                analysis_href = analysis_td.find('a').get('href') if (analysis_td and analysis_td.find('a')) else ''
                analysis_url = f"{self.base_url}{analysis_href}" if analysis_href else ''

                data_list.append({
                    '比赛id': match_id,
                    '赛事': league,
                    '时间': match_time,
                    '主队': home_team,
                    '客队': away_team,
                    '全场主队得分': home_score,
                    '全场客队得分': away_score,
                    '主胜SP值': sp_win,
                    '主平SP值': sp_draw,
                    '主负SP值': sp_lose,
                    '分析链接': analysis_url
                })
            except Exception as e:
                logging.warning(f"解析单行数据失败 (ID: {row.get('matchid')}): {e}")
                continue

        return data_list

    def get_data(self, date):
        """主入口：输入日期，返回 DataFrame"""
        html = self.fetch_html(date)
        if not html:
            return pd.DataFrame()

        data = self.parse_html(html)
        df = pd.DataFrame(data)
        df["期数id"] = date
        cols = df.columns.tolist()
        new_cols = [cols[-1]] + cols[:-1]

        if not df.empty:
            logging.info(f"数据采集完成，共 {len(df)} 条记录")
        else:
            logging.warning("未采集到任何数据")

        return df[new_cols]

# --- 使用示例 ---s
if __name__ == "__main__":
    scraper = OkoooScraper()
    
    # 获取项目根目录 (source 的上一级)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    overview_dir = os.path.join(project_root, 'data', 'overview')
    
    # 确保目录存在
    os.makedirs(overview_dir, exist_ok=True)
    
    # 输入你需要的 date 参数
    for i in range(26001, 26210):
        file_path = os.path.join(overview_dir, f'{i}.csv')
        if os.path.exists(file_path):
            logging.info(f"文件 {file_path} 已存在，跳过爬取。")
            continue

        result_df = scraper.get_data(date=f"{i}")

        # 打印前 5 行查看结果
        # print("\n--- 结果预览 ---")
        # print(result_df.head())

        if not result_df.empty:
        # 如果需要保存
            result_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            break
            # continue

        time.sleep(1)  # 等待1秒