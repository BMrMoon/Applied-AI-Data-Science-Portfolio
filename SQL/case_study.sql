SELECT * FROM "Customers"

-- 	2.	Write the query that shows how many distinct customers made a purchase.
SELECT COUNT(DISTINCT("master_id")) FROM "Customers"

-- 	3.	Write the query that returns the total number of purchases and total revenue.
SELECT COUNT(master_id) AS TOTAL_SALE, (SUM(customer_value_total_ever_offline) + SUM(customer_value_total_ever_online)) AS CIRO FROM "Customers"

-- 	4.	Write the query that returns the average revenue per purchase.
SELECT (SUM(customer_value_total_ever_offline) + SUM(customer_value_total_ever_online))/COUNT(master_id) AS AVERAGE_CIRO FROM "Customers"

-- 	5.	Write the query that returns the total revenue and number of purchases made through the last purchase channel (last_order_channel).
SELECT 
	last_order_channel,
	(SUM(customer_value_total_ever_offline) + SUM(customer_value_total_ever_online)) AS CIRO,
	COUNT(master_id) AS TOTAL_SALE
FROM "Customers"
GROUP BY last_order_channel

-- 	6.	Write the query that returns the total revenue broken down by store type.
SELECT 
	store_type,
	ROUND((SUM(customer_value_total_ever_offline) + SUM(customer_value_total_ever_online))) AS CIRO
FROM "Customers"
GROUP BY store_type

-- 7.	Write the query that returns the number of purchases by year (use the year of the customer’s first purchase date, first_order_date, as the basis for the year).
SELECT
    CASE
        WHEN EXTRACT(YEAR FROM AGE(first_order_date)) >= 7 THEN '7_year_and_greater'
        ELSE 'Less_then_7_year'
    END AS year_breakdown,
    COUNT(master_id) customer_number
FROM "Customers"
GROUP BY
    CASE
        WHEN EXTRACT(YEAR FROM AGE(first_order_date)) >= 7 THEN '7_year_and_greater'
        ELSE 'Less_then_7_year'
    END
ORDER BY year_breakdown;

-- 8.	Write the query that calculates the average revenue per purchase, broken down by the last purchase channel.
SELECT last_order_channel, AVG((customer_value_total_ever_offline)+(customer_value_total_ever_online))
FROM "Customers"
GROUP BY last_order_channel

-- 	9.	Write the query that returns the most popular category in the last 12 months.
SELECT
    TRIM(category) AS categories,
	COUNT(*) AS COUNTS
FROM "Customers",
LATERAL unnest(
    STRING_TO_ARRAY(
        REPLACE(REPLACE(interested_in_categories_12, '[', ''), ']', ''),
        ','
    )
) AS category
WHERE last_order_date >= (
    SELECT MAX(last_order_date) - INTERVAL '12 months'
    FROM "Customers"
)
GROUP BY TRIM(category)
ORDER BY COUNTS DESC
LIMIT 1;

-- 	10.	Write the query that returns the most preferred store_type.
SELECT store_type
FROM "Customers"
GROUP BY store_type
ORDER BY COUNT(master_id) DESC
LIMIT 1;

-- 	11.	Write the query that returns, based on the last purchase channel (last_order_channel), the most popular category and the total amount purchased from that category.
SELECT last_order_channel, TRIM(category), COUNT(TRIM(category))
FROM "Customers",
LATERAL unnest(
    STRING_TO_ARRAY(
        REPLACE(REPLACE(interested_in_categories_12, '[', ''), ']', ''),
        ','
    )
) AS category
GROUP BY last_order_channel, TRIM(category)
ORDER BY COUNT(TRIM(category)) DESC;

SELECT channel, categories, counts
FROM (
	SELECT 
		channel,
		categories,
		counts,
		ROW_NUMBER() OVER (
			PARTITION BY channel
			ORDER BY counts DESC
		) AS ranks
	FROM (
		SELECT
			last_order_channel AS channel,
			TRIM(category) AS categories,
			COUNT(TRIM(category)) AS counts
		FROM "Customers",
		LATERAL unnest(
    		STRING_TO_ARRAY(
        		REPLACE(REPLACE(interested_in_categories_12, '[', ''), ']', ''),
        		','
			)
		) AS category
		GROUP BY last_order_channel, TRIM(category)
	) t
) x
WHERE ranks = 1;

-- 	12.	Write the query that returns the ID of the customer who made the highest number of purchases.
SELECT master_id
FROM "Customers"
GROUP BY master_id
ORDER BY SUM(customer_value_total_ever_offline) + SUM(customer_value_total_ever_online) DESC
LIMIT 1

-- 	13.	Write the query that returns the average revenue per purchase and the average number of days between purchases (purchase frequency) for the customer who made the highest number of purchases.
SELECT AVG(customer_value_total_ever_offline + customer_value_total_ever_online)
FROM "Customers"
GROUP BY master_id
ORDER BY SUM(customer_value_total_ever_offline) + SUM(customer_value_total_ever_online) DESC
LIMIT 1

-- 	14.	Write the query that returns the average number of days between purchases (purchase frequency) for the top 100 customers who made the highest amount of purchases in terms of revenue.
SELECT (last_order_date::date - first_order_date::date)/(order_num_total_ever_online + order_num_total_ever_offline) AS buy_frequency
FROM "Customers"
ORDER BY (SELECT SUM(customer_value_total_ever_offline) + SUM(customer_value_total_ever_online) FROM "Customers") DESC
LIMIT 100;


-- 	15.	Write the query that returns the customer who made the highest number of purchases for each last purchase channel (last_order_channel).
SELECT channel, master_id, counts
FROM (
	SELECT 
		channel,
		master_id,
		counts,
		ROW_NUMBER() OVER (
			PARTITION BY channel
			ORDER BY counts DESC
		) AS ranks
	FROM (
		SELECT
			last_order_channel AS channel,
			master_id,
			order_num_total_ever_online + order_num_total_ever_offline AS counts
		FROM "Customers"
	) x
) t
WHERE ranks=1

-- 	16.	Write the query that returns the ID of the customer who made the most recent purchase. (There are multiple customer IDs with purchases on the latest date. Return all of them as well.)
SELECT *
FROM "Customers"
WHERE last_order_date = (SELECT MAX(last_order_date) FROM "Customers");

