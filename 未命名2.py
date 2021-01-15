from selenium import webdriver
from lxml import etree
import time
import xlsxwriter

driver = webdriver.Chrome()
driver.maximize_window()
url = 'https://gongyi.qq.com/succor/project_list.htm#s_status=3&s_tid=71&p=665'
workbook = xlsxwriter.Workbook('腾讯公益平台18.xlsx')    # 创建一个名为‘腾讯公益案例1.xlsx’的工作表
sheet = workbook.add_worksheet()            # 创建一个工作表对象
row = 1
row0 = [u'简介',u'项目介绍',u'目标金额',u'已筹金额',u'捐款人数',u'发起机构',u'公益机构']
# 生成第一行
sheet.write_row('A1',row0)

def enter_Project(the_url, number, page,row):
    driver.get(the_url)
    driver.implicitly_wait(10)
    # get_info(driver.current_url)
    try:
        driver.find_elements_by_class_name('titless')[int(number)].click()
        number = number + 1
    except:
        enterNextPage(driver.current_url, number, page, row)
    driver.implicitly_wait(10)
    if (number == 10):
        enterNextPage(driver.current_url, number, page, row)
    else:
        driver.switch_to.window(driver.window_handles[-1])  # 页面跳转
        get_info(driver.current_url, number, page, row)


def get_info(this_url, number, page, row):
    driver.get(this_url)
    selector = etree.HTML(driver.page_source)
    driver.implicitly_wait(10)
    tittles = selector.xpath('//span[@id="pj_name"]//text()')  # 标题
    introduction = [''.join(selector.xpath('//div[@id="pj_content"]/p//text()'))]  # 项目介绍
    targetAmount = selector.xpath('//*[@id="money_already"]//text()')  # 目标金额
    raisedMoney = selector.xpath('//*[@id="target_span"]//text()')  # 已筹金额
    numberOfDonations = selector.xpath('//*[@id="project_donateNum"]//text()')  # 捐献人次
    sponsors = selector.xpath(
        '//*[@id="fund_info_wrap"]/div[@class="main_bottom_right_organization_info clearfix"]/dl[@class="p-e-name"]/dd/text()')  # 发起机构
    publicSupport = [selector.xpath(
        'normalize-space(//*[@id="fund_info_wrap"]/div[@class="main_bottom_right_organization_info clearfix"  ]/dl[@class="fundName-dl"]/dt[@class="fundName-dt"]/text())')]  # 公益机构
    # print('下面是爬取的数据：')
    myList = [tittles, introduction, targetAmount, raisedMoney, numberOfDonations, sponsors, publicSupport]
    #print(myList, row)
    for i in range(0, len(myList)):
        if (myList[i] == []):
            sheet.write(row, i, '')
        else:
            sheet.write(row, i, myList[i][0])
    print(row)
    row = row + 1
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    driver.implicitly_wait(10)
    if (number <= 10):
        #print(number)
        enter_Project(driver.current_url, number, page, row)


def enterNextPage(url, number, page, row):
    driver.get(url)
    driver.implicitly_wait(10)
    if (page < 2):
        driver.implicitly_wait(10)
        # driver.execute_script("window.scrollBy(0, 1000)")
        driver.find_element_by_xpath('//*[@id="projectPages_wrap"]/a[6]').click()
        driver.implicitly_wait(10)
        page = page + 1
        enter_Project(driver.current_url, 0, page, row)
        # print(page)
    elif (page >= 11):
        pass
    else:
        driver.implicitly_wait(10)
        # driver.execute_script("window.scrollBy(0, 1000)")
        driver.find_element_by_xpath('//*[@id="projectPages_wrap"]/a[7]').click()
        driver.implicitly_wait(10)
        page = page + 1
        enter_Project(driver.current_url, 0, page, row)
        # print(page)


if __name__ == '__main__':
    driver.get(url)
    driver.implicitly_wait(10)
    enter_Project(url, 0, 1, row)
    workbook.close()  # 关闭Excel文件
    time.sleep(10)
    driver.quit()