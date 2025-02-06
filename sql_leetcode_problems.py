''' 3436. Find Valid Emails ''' 
# Write your MySQL query statement below
select user_id, 
    email
from users
where email like  '%.com'
and email regexp '^[a-zA-Z0-9_]+@[a-zA-Z]+\\.com$'
order by user_id ; 

''' 1729. Find Followers Count ''' 
# Write your MySQL query statement below
select user_id, 
    count(follower_id) as followers_count
from followers
group by user_id
order by user_id ; 


''' 1731. The Number of Employees Which Report to Each Employee ''' 
with temp_temp as (
    select e1.employee_id, 
    e1.name,
    count(e2.reports_to) as reports_count, 
    round(avg(e2.age)) as average_age
    from employees e1 
    inner join employees e2  on e1.employee_id = e2.reports_to 
    group by e1.employee_id 
)
select * 
from temp_temp order by employee_id ; 


''' Problem 1: Complex Fraud Detection with Recursive Analysis (Very Hard)
Description
You are given a table tracking financial transactions. 
Fraudulent transactions are identified based on circular money movement—if a user sends money,
and after a series of transactions, the money comes back to them, it’s flagged as fraud. 
You must identify all users involved in circular transactions.
''' 

with recursive cycle_detection AS (
	select sender_id  as start_user, 
		receiver_id , 
		amount, 
		transaction_time, 
		1 as depth , 
		sender_id as cycle_path

	from transactions 
	where transaction_time between '2024-01-01 00:01:00' and  '2024-01-01 23:59:59' 

	union all 

	select 
		c.start_user, 
		t.receiver_id, 
		t.amount, 
		t.transaction_time, 
		c.depth + 1, 
		concat(c.cycle_path,'->',t.receiver_id) as cycle_path 

	from cycle_detection c 
	join transactions t  on c.receiver_id = t.sender_id 
		and t.transaction_time between c.transaction_time and c.transaction_time + interval 24 hour
	where c.depth < 5 
	) 
select distinct start_user as user_id 
from cycle_detection 
where start_user = receiver_id
and depth >= 3 ; 

